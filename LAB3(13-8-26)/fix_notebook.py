import json

with open("ILLM_Lab3_Gujarati_BPE_WordPiece.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find and replace in cells
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        source = cell["source"]
        source_str = "".join(source)
        
        # BPE Train
        if "class BPETokenizer:" in source_str:
            new_train_bpe = """    def train(self, sentences: List[str]):
        t0 = time.time()
        word_freqs_raw = self._get_word_freqs(sentences)
        
        chars = set()
        for word in word_freqs_raw:
            for ch in word:
                chars.add(ch)
        self.vocab = {ch: i for i, ch in enumerate(sorted(chars))}
        self.merges = []
        num_merges = max(0, self.vocab_size - len(self.vocab))
        
        word_list = [list(w) for w in word_freqs_raw.keys()]
        word_counts = list(word_freqs_raw.values())
        
        pair_freqs = Counter()
        pair_to_words = defaultdict(set)
        
        for idx, word in enumerate(word_list):
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                pair_freqs[pair] += word_counts[idx]
                pair_to_words[pair].add(idx)

        for step in range(num_merges):
            if not pair_freqs:
                break
            best_pair = pair_freqs.most_common(1)[0][0]
            merged_symbol = best_pair[0] + best_pair[1]
            self.vocab[merged_symbol] = len(self.vocab)
            self.merges.append(best_pair)
            
            words_to_update = list(pair_to_words[best_pair])
            del pair_freqs[best_pair]
            del pair_to_words[best_pair]

            for idx in words_to_update:
                word = word_list[idx]
                count = word_counts[idx]

                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    if pair != best_pair:
                        pair_freqs[pair] -= count
                        pair_to_words[pair].discard(idx)
                        if pair_freqs[pair] <= 0:
                            del pair_freqs[pair]

                new_word = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                        new_word.append(merged_symbol)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                word_list[idx] = new_word

                for i in range(len(new_word) - 1):
                    pair = (new_word[i], new_word[i + 1])
                    pair_freqs[pair] += count
                    pair_to_words[pair].add(idx)
            
            if (step + 1) % 2000 == 0 or step == num_merges - 1:
                print(f"    [BPE] Merge {step+1}/{num_merges} | Vocab Size: {len(self.vocab):,} | Elapsed: {time.time()-t0:.1f}s")
                
        print(f"  -> BPE Training Complete. Final Vocab: {len(self.vocab):,} subwords ({time.time()-t0:.1f}s)")
"""
            # Replace train method
            import re
            source_str = re.sub(r'    def train\(self, sentences: List\[str\]\):.*?(?=\n    def encode\()', new_train_bpe + '\n', source_str, flags=re.DOTALL)
            # update cell
            cell["source"] = [line + "\n" for line in source_str.split("\n")]
            if cell["source"][-1] == "\n":
                cell["source"] = cell["source"][:-1]
            else:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")

        # WordPiece Train and Encode
        if "class WordPieceTokenizer:" in source_str:
            new_train_wp = """    def train(self, sentences: List[str]):
        t0 = time.time()
        word_freqs_raw = self._get_word_freqs(sentences)
        
        char_set = set()
        for word in word_freqs_raw:
            char_set.add(word[0])
            for ch in word[1:]:
                char_set.add(self.CONT_PREFIX + ch)
                
        self.vocab = {self.UNK: 0}
        for i, ch in enumerate(sorted(char_set), 1):
            self.vocab[ch] = i
            
        word_list = []
        word_counts = []
        for word, count in word_freqs_raw.items():
            split = [word[0]] + [self.CONT_PREFIX + c for c in word[1:]]
            word_list.append(split)
            word_counts.append(count)

        pair_freqs = Counter()
        symbol_freqs = Counter()
        pair_to_words = defaultdict(set)

        for idx, split in enumerate(word_list):
            count = word_counts[idx]
            for s in split:
                symbol_freqs[s] += count
            for i in range(len(split) - 1):
                pair = (split[i], split[i + 1])
                pair_freqs[pair] += count
                pair_to_words[pair].add(idx)
            
        num_merges = max(0, self.vocab_size - len(self.vocab))
        
        for step in range(num_merges):
            if not pair_freqs:
                break
                
            best_pair = None
            best_score = -1.0
            for pair, freq in pair_freqs.items():
                denom = symbol_freqs[pair[0]] * symbol_freqs[pair[1]]
                if denom == 0:
                    continue
                score = freq / denom
                if score > best_score:
                    best_score = score
                    best_pair = pair
                    
            if best_pair is None:
                break
                
            merged = best_pair[0] + best_pair[1][2:] if best_pair[1].startswith(self.CONT_PREFIX) else best_pair[0] + best_pair[1]
            self.vocab[merged] = len(self.vocab)
            
            words_to_update = list(pair_to_words[best_pair])
            del pair_freqs[best_pair]
            del pair_to_words[best_pair]

            for idx in words_to_update:
                split = word_list[idx]
                count = word_counts[idx]

                for s in split:
                    symbol_freqs[s] -= count
                for i in range(len(split) - 1):
                    pair = (split[i], split[i + 1])
                    if pair != best_pair:
                        pair_freqs[pair] -= count
                        pair_to_words[pair].discard(idx)
                        if pair_freqs[pair] <= 0:
                            del pair_freqs[pair]

                new_split = []
                i = 0
                while i < len(split):
                    if i < len(split) - 1 and split[i] == best_pair[0] and split[i + 1] == best_pair[1]:
                        new_split.append(merged)
                        i += 2
                    else:
                        new_split.append(split[i])
                        i += 1
                word_list[idx] = new_split
                
                for s in new_split:
                    symbol_freqs[s] += count
                for i in range(len(new_split) - 1):
                    pair = (new_split[i], new_split[i + 1])
                    pair_freqs[pair] += count
                    pair_to_words[pair].add(idx)
                
            if (step + 1) % 2000 == 0 or step == num_merges - 1:
                print(f"    [WordPiece] Merge {step+1}/{num_merges} | Vocab Size: {len(self.vocab):,} | Elapsed: {time.time()-t0:.1f}s")
                
            if len(self.vocab) >= self.vocab_size:
                break
                
        print(f"  -> WordPiece Training Complete. Final Vocab: {len(self.vocab):,} subwords ({time.time()-t0:.1f}s)")
"""
            import re
            source_str = re.sub(r'    def train\(self, sentences: List\[str\]\):.*?(?=\n    def encode\()', new_train_wp + '\n', source_str, flags=re.DOTALL)
            
            # fix wordpiece truncation bug
            source_str = re.sub(
                r'                if not found:\n                    sub_tokens\.append\(self\.UNK\)\n                    break',
                r'                if not found:\n                    sub_tokens = [self.UNK]\n                    break',
                source_str
            )
            
            cell["source"] = [line + "\n" for line in source_str.split("\n")]
            if cell["source"][-1] == "\n":
                cell["source"] = cell["source"][:-1]
            else:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")

with open("ILLM_Lab3_Gujarati_BPE_WordPiece.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Notebook updated successfully.")
