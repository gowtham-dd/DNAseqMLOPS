from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from collections import Counter
import math
import os

class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.X = None  # Stores features in memory (like original notebook)
        self.y = None  # Stores labels in memory (like original notebook)
        
    def _nucleotide_composition(self, seq):
        """EXACTLY same as original notebook"""
        return {
            'length': len(seq),
            'A_perc': seq.count('A')/len(seq),
            'C_perc': seq.count('C')/len(seq),
            'G_perc': seq.count('G')/len(seq),
            'T_perc': seq.count('T')/len(seq),
            'GC_content': (seq.count('G')+seq.count('C'))/len(seq)
        }

    def _get_kmers(self, sequence, k=3):
        """EXACTLY same as original notebook"""
        return [sequence[i:i+k] for i in range(len(sequence) - k + 1)]

    def _shannon_entropy(self, seq):
        """EXACTLY same as original notebook"""
        counts = Counter(seq)
        probs = [c/len(seq) for c in counts.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    def transform(self):
        """Only transforms if output folder doesn't exist"""
        if not os.path.exists(self.config.root_dir):
            os.makedirs(self.config.root_dir)
            print(f"Transforming data (folder didn't exist)...")
            
            # EXACTLY same processing as original notebook
            df = pd.read_csv(self.config.data_path)
            self.y = df['Y'].values
            
            # 1. Nucleotide composition
            comp_features = df['DNA'].apply(self._nucleotide_composition).apply(pd.Series)
            
            # 2. K-mer features
            vectorizer = CountVectorizer(analyzer='char', ngram_range=(3,3), max_features=500)
            X_kmer = vectorizer.fit_transform(df['DNA'].apply(lambda x: ' '.join(self._get_kmers(x,3))))
            kmer_features = pd.DataFrame(X_kmer.toarray(), 
                                       columns=[f"3mer_{name}" for name in vectorizer.get_feature_names_out()])
            
            # 3. Complexity features
            complexity_features = df['DNA'].apply(lambda x: pd.Series({
                'entropy': self._shannon_entropy(x),
                'unique_kmers': len(set(self._get_kmers(x,3))),
                'repeats': len(x) - len(set(x))
            }))
            
            # Combine and select features
            all_features = pd.concat([comp_features, kmer_features, complexity_features], axis=1)
            selector = SelectKBest(chi2, k=100)
            self.X = selector.fit_transform(all_features, self.y)
            
            # Save to memory (no file writing)
            print("Transformation complete - results stored in memory")
            return True
        else:
            print(f"Folder {self.config.root_dir} exists - skipping transformation")
            return False

    def get_features(self):
        """Returns features exactly as original notebook would have them"""
        return self.X, self.y