import os
import time
import re
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nltk
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, ARMAConv, SGConv

# For visualizations and evaluation
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize

# Ensure WordNet is available
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

# -------------------- Data Loading and Preprocessing -------------------- #
def clean_text(text):
    """Lowercase and remove punctuation from text."""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def load_data(source_file, label_file):
    """Load tweets and labels from files, cleaning the text."""
    tweet_ids, tweets = [], []
    with open(source_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            tweet_ids.append(parts[0])
            tweets.append(clean_text(parts[1]))
    labels_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                label, tweet_id = line.split(':', 1)
                labels_dict[tweet_id] = label.lower()
            except Exception as e:
                print("Error parsing line:", line, e)
    filtered_tweet_ids, filtered_tweets, filtered_labels = [], [], []
    for tid, tweet in zip(tweet_ids, tweets):
        if tid in labels_dict:
            filtered_tweet_ids.append(tid)
            filtered_tweets.append(tweet)
            filtered_labels.append(labels_dict[tid])
    return filtered_tweet_ids, filtered_tweets, filtered_labels

def get_synonyms(word):
    """Return synonyms for a word using WordNet."""
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonym = lemma.name().replace('_', ' ')
            if synonym != word and synonym not in synonyms:
                synonyms.append(synonym)
    return synonyms

def synonym_replacement(text, n=1):
    """Replace n words in text with one of their synonyms."""
    words = text.split()
    if len(words) <= 1:
        return text
    words_with_synonyms = [word for word in words if len(get_synonyms(word)) > 0]
    if not words_with_synonyms:
        return text
    n = min(n, len(words_with_synonyms))
    words_to_replace = random.sample(words_with_synonyms, n)
    new_words = []
    for word in words:
        if word in words_to_replace:
            syns = get_synonyms(word)
            new_words.append(random.choice(syns) if syns else word)
        else:
            new_words.append(word)
    return ' '.join(new_words)

def perform_data_augmentation(tweets, labels):
    """Augment under-represented classes using synonym replacement."""
    augmented_tweets, augmented_labels = [], []
    label_counts = {label: labels.count(label) for label in set(labels)}
    max_count = max(label_counts.values())
    for label in label_counts:
        if label_counts[label] >= max_count * 0.9:
            continue
        indices = [i for i, l in enumerate(labels) if l == label]
        num_to_augment = min(int(max_count * 0.9) - label_counts[label], len(indices) * 2)
        for _ in range(num_to_augment):
            idx = random.choice(indices)
            augmented_tweet = synonym_replacement(tweets[idx], n=random.randint(1, 3))
            augmented_tweets.append(augmented_tweet)
            augmented_labels.append(label)
    all_tweets = tweets + augmented_tweets
    all_labels = labels + augmented_labels
    return all_tweets, all_labels

def create_enhanced_features(tweets):
    """Extract TF-IDF features with advanced n-gram settings."""
    vectorizer = TfidfVectorizer(
        max_features=2500,
        ngram_range=(1, 3),
        min_df=2,
        max_df=0.9,
        sublinear_tf=True,
        use_idf=True,
        norm='l2'
    )
    X = vectorizer.fit_transform(tweets)
    X_dense = X.toarray()
    features = torch.tensor(X_dense, dtype=torch.float)
    return features, vectorizer

def create_robust_edge_index(features, k=5):
    """Create a k-nearest neighbors graph using cosine similarity."""
    X_dense = features.numpy()
    cosine_sim = cosine_similarity(X_dense)
    num_nodes = cosine_sim.shape[0]
    edge_list = []
    for i in range(num_nodes):
        top_indices = np.argsort(cosine_sim[i])[::-1][1:k+1]
        for j in top_indices:
            edge_list.append((i, j))
    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

def drop_edge(edge_index, drop_rate):
    """Randomly drop edges for regularization."""
    if drop_rate <= 0:
        return edge_index
    num_edges = edge_index.size(1)
    mask = torch.rand(num_edges, device=edge_index.device) > drop_rate
    return edge_index[:, mask]

# -------------------- GNN Model Definitions -------------------- #
# 1. BiGCN_A: Two-layer GCN
class BiGCN_A(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super(BiGCN_A, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 2. BiGCN_B: Three-layer GCN (with an extra hidden layer)
class BiGCN_B(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super(BiGCN_B, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, num_classes)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index)
        return x

# 3. BiGAT: Two-layer Graph Attention Network
class BiGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5, heads=8):
        super(BiGAT, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True, dropout=dropout)
        self.conv2 = GATConv(hidden_channels * heads, num_classes, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 4. BiSAGE: Two-layer GraphSAGE
class BiSAGE(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5):
        super(BiSAGE, self).__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, num_classes)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 5. BiARMA: Two-layer ARMA convolution network
class BiARMA(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes, dropout=0.5, num_stacks=1, num_layers=1):
        super(BiARMA, self).__init__()
        self.conv1 = ARMAConv(in_channels, hidden_channels, num_stacks=num_stacks, num_layers=num_layers,
                              shared_weights=True, dropout=dropout)
        self.conv2 = ARMAConv(hidden_channels, num_classes, num_stacks=num_stacks, num_layers=num_layers,
                              shared_weights=True, dropout=dropout)
        self.dropout = dropout
    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

# 6. BiSGCN: Simplified GCN using SGConv
class BiSGCN(nn.Module):
    def __init__(self, in_channels, num_classes, K=2):
        super(BiSGCN, self).__init__()
        self.conv = SGConv(in_channels, num_classes, K=K)
    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        return x

# -------------------- Training and Evaluation -------------------- #
def train_model(model, data, labels, train_mask=None, val_mask=None, num_epochs=200, patience=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    data = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50)
    labels_np = np.array(labels)
    if train_mask is None or val_mask is None:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        for train_index, val_index in sss.split(np.zeros(len(labels_np)), labels_np):
            train_mask = torch.tensor(train_index, dtype=torch.long)
            val_mask = torch.tensor(val_index, dtype=torch.long)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    best_model_state = None
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[train_mask], torch.tensor(labels_np[train_mask.cpu().numpy()], device=device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.eval()
        with torch.no_grad():
            val_out = model(data.x, data.edge_index)
            val_loss = F.cross_entropy(val_out[val_mask], torch.tensor(labels_np[val_mask.cpu().numpy()], device=device))
        train_losses.append(loss.item())
        val_losses.append(val_loss.item())
        if val_loss.item() < best_val_loss:
            best_val_loss = val_loss.item()
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    model.load_state_dict(best_model_state)
    return model, train_mask, val_mask, train_losses, val_losses

def test_model(model, data, test_mask, true_labels):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with torch.no_grad():
        out = model(data.x.to(device), data.edge_index.to(device))
        prob = F.softmax(out, dim=1)
        pred = out[test_mask].max(1)[1]
        acc = (pred.cpu().numpy() == true_labels).mean()
    test_prob = prob[test_mask].cpu().numpy()
    return acc, test_prob, pred.cpu().numpy()

# -------------------- Visualization Functions -------------------- #
def plot_loss_curves(train_losses, val_losses, title="Loss Curves"):
    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.legend()
    plt.savefig(f"{title}.png")
    plt.show()

def plot_conf_matrix(true_labels, pred_labels, classes, title="Confusion Matrix"):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(f"{title}.png")
    plt.show()

def plot_roc_curves(true_labels, test_prob, num_classes, title="ROC Curves"):
    true_binarized = label_binarize(true_labels, classes=range(num_classes))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(true_binarized[:, i], test_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    plt.figure()
    for i in range(num_classes):
        plt.plot(fpr[i], tpr[i], label=f"Class {i} (AUC = {roc_auc[i]:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(f"{title}.png")
    plt.show()

# -------------------- Main Pipeline -------------------- #
def main():
    # Set file paths (adjust as necessary)
    source_file = 'twitter_data/twitter16/twitter16_source_tweets.txt'
    label_file = 'twitter_data/twitter16/twitter16_label.txt'
    
    print("Starting pipeline...")
    tweet_ids, tweets, labels = load_data(source_file, label_file)
    print(f"Loaded {len(tweets)} tweets.")
    tweets, labels = perform_data_augmentation(tweets, labels)
    print("Data augmentation completed.")
    features, vectorizer = create_enhanced_features(tweets)
    print("Feature extraction completed.")
    edge_index = create_robust_edge_index(features, k=5)
    print("Edge index created.")
    data = Data(x=features, edge_index=edge_index)
    
    # Map labels to integers
    label_mapping = {lbl: i for i, lbl in enumerate(sorted(set(labels)))}
    labels_int = [label_mapping[lbl] for lbl in labels]
    
    labels_np = np.array(labels_int)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    for train_index, test_index in sss.split(np.zeros(len(labels_np)), labels_np):
        fixed_train_mask = torch.tensor(train_index, dtype=torch.long)
        fixed_test_mask = torch.tensor(test_index, dtype=torch.long)
    
    architectures = {
        "BiGCN_A": lambda in_c, hid, out: BiGCN_A(in_c, hid, out, dropout=0.5),
        "BiGCN_B": lambda in_c, hid, out: BiGCN_B(in_c, hid, out, dropout=0.5),
        "BiGAT": lambda in_c, hid, out: BiGAT(in_c, hid, out, dropout=0.5, heads=8),
        "BiSAGE": lambda in_c, hid, out: BiSAGE(in_c, hid, out, dropout=0.5),
        "BiARMA": lambda in_c, hid, out: BiARMA(in_c, hid, out, dropout=0.5, num_stacks=1, num_layers=1),
        "BiSGCN": lambda in_c, hid, out: BiSGCN(in_c, out, K=2)
    }
    
    results = {}
    in_channels = features.shape[1]
    num_classes = len(label_mapping)
    hidden_channels = 128
    
    for arch_name, model_fn in architectures.items():
        print(f"\nTraining architecture: {arch_name}")
        model = model_fn(in_channels, hidden_channels, num_classes)
        model, train_mask, val_mask, train_losses, val_losses = train_model(model, data, labels_int,
                                                  train_mask=fixed_train_mask,
                                                  val_mask=fixed_test_mask,
                                                  num_epochs=200, patience=20)
        acc, test_prob, test_pred = test_model(model, data, fixed_test_mask, labels_np[fixed_test_mask.cpu().numpy()])
        results[arch_name] = {
            "accuracy": acc,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "test_prob": test_prob,
            "test_pred": test_pred,
            "test_true": labels_np[fixed_test_mask.cpu().numpy()]
        }
        print(f"{arch_name} Test Accuracy: {acc*100:.2f}%")
    
    # Bar chart comparing test accuracies
    arch_names = list(results.keys())
    accuracies = [results[a]["accuracy"] * 100 for a in arch_names]
    plt.figure(figsize=(10, 6))
    plt.bar(arch_names, accuracies, color='skyblue')
    plt.xlabel("GNN Architecture")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Comparison of GNN Architectures for Fake News Detection")
    plt.ylim(0, 100)
    plt.savefig("twitter16_gnn_architecture_comparison.png")
    plt.show()
    
    # Define class labels from label_mapping
    classes = [str(lbl) for lbl in sorted(label_mapping, key=lambda x: label_mapping[x])]
    for arch_name in arch_names:
        print(f"\nVisualizations for {arch_name}")
        plot_loss_curves(results[arch_name]["train_losses"], results[arch_name]["val_losses"],
                         title=f"{arch_name} Loss Curves")
        plot_conf_matrix(results[arch_name]["test_true"], results[arch_name]["test_pred"],
                         classes, title=f"{arch_name} Confusion Matrix")
        plot_roc_curves(results[arch_name]["test_true"], results[arch_name]["test_prob"],
                        num_classes, title=f"{arch_name} ROC Curves")
    
if __name__ == '__main__':
    main()