def preprocess(df):
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    text = [str(t) for t in df['text']]
    tokens = tokenizer(text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return tokens.input_ids, tokens.attention_mask, labels


train_input_ids, train_attention_mask, train_labels = preprocess(train_df)
test_input_ids, test_attention_mask, test_labels = preprocess(test_df)

train_data = TensorDataset(train_input_ids, train_attention_mask, train_labels)
train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE, num_workers=4)

test_data = TensorDataset(test_input_ids, test_attention_mask, test_labels)
test_dataloader = DataLoader(test_data, sampler=SequentialSampler(test_data), batch_size=BATCH_SIZE, num_workers=4)

word2vec_model = KeyedVectors.load_word2vec_format('expanded_model_GLOVE.bin', binary=True)
vocab_size, embedding_dim = len(word2vec_model.index_to_key), word2vec_model.vector_size

word_embedding = torch.zeros(vocab_size, embedding_dim, device=device)
word2index = {word: idx for idx, word in enumerate(word2vec_model.index_to_key)}
index2word = {idx: word for word, idx in word2index.items()}

for idx, word in enumerate(word2vec_model.index_to_key):
    word_embedding[idx] = torch.tensor(word2vec_model[word], device=device)


def replace_unknown_ids(input_ids, word_embedding):
    notinvec = 0
    replaced_input_ids = input_ids.clone()

    for i in range(input_ids.size(0)):
        for j in range(input_ids.size(1)):
            token = tokenizer.decode([input_ids[i, j].item()])
            if token in word2index:
                replaced_input_ids[i, j] = word2index[token]
            elif token == '<pad>':
                pad_index = word2index.setdefault('<pad>', len(word2index))
                if pad_index == len(word_embedding):
                    zero_vector = torch.zeros(1, embedding_dim).to(device)
                    word_embedding = torch.cat([word_embedding, zero_vector], dim=0)
                replaced_input_ids[i, j] = pad_index
            else:
                unk_index = word2index.setdefault(token, len(word2index))
                if unk_index == len(word_embedding):
                    random_vector = (torch.rand(1, embedding_dim) * 5).to(device)
                    word_embedding = torch.cat([word_embedding, random_vector], dim=0)
                    notinvec += 1
                replaced_input_ids[i, j] = unk_index
    return replaced_input_ids, word_embedding, notinvec


class QueryMapping(nn.Module):
    def __init__(self):
        super(QueryMapping, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(128, 1024)

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class SensoryT5Model(nn.Module):
    def __init__(self, num_labels, word_embedding):
        super(SensoryT5Model, self).__init__()
        self.t5 = T5Model.from_pretrained("t5-large")
        self.word_embedding = word_embedding
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16)
        self.query_mapping = QueryMapping()
        self.classifier = nn.Linear(1024, num_labels)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_ids, decoder_input_ids, labels):
        new_input_ids, self.word_embedding, _ = replace_unknown_ids(input_ids, self.word_embedding)
        t5_output = self.t5(input_ids=input_ids, decoder_input_ids=decoder_input_ids, return_dict=True,
                            output_hidden_states=True)
        last_hidden_state = t5_output.decoder_hidden_states[-1]
        t5_k = self.t5.decoder.block[-1].layer[0].SelfAttention.k(last_hidden_state)

        selected_word_embeddings = self.word_embedding[new_input_ids]
        custom_query = self.query_mapping(selected_word_embeddings.transpose(0, 1))

        attn_output, _ = self.attn(custom_query, t5_k.transpose(0, 1), last_hidden_state.transpose(0, 1))
        final_hidden_state = attn_output * last_hidden_state.transpose(0, 1).transpose(0, 1)

        logits = self.log_softmax(self.classifier(final_hidden_state[:, 0]))
        if labels is not None:
            loss = nn.NLLLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}