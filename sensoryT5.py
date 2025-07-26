#Data prep
def preprocess(df):
    labels = torch.tensor(df['label'].values, dtype=torch.long)
    text   = [str(t) for t in df['text']]
    tokens = tokenizer(text, padding=True, truncation=True, max_length=MAX_LEN, return_tensors="pt")
    return tokens.input_ids, tokens.attention_mask, labels

train_input_ids, train_attention_mask, train_labels = preprocess(train_df)
test_input_ids,  test_attention_mask,  test_labels  = preprocess(test_df)

train_data = TensorDataset(train_input_ids, train_attention_mask, train_labels)
test_data  = TensorDataset(test_input_ids,  test_attention_mask,  test_labels)

train_dataloader = DataLoader(train_data, sampler=RandomSampler(train_data), batch_size=BATCH_SIZE, num_workers=4)
test_dataloader  = DataLoader(test_data,  sampler=SequentialSampler(test_data), batch_size=BATCH_SIZE,  num_workers=4)

#Build a static T5-id → w2v-id map
PAD_ID = tokenizer.pad_token_id
embedding_dim = word2vec_model.vector_size
base_w2v = torch.tensor(word2vec_model.vectors, dtype=torch.float32, device=device)
word2index = word2vec_model.key_to_index
orig_vocab_size = base_w2v.size(0)

id2w2v_list = []
extra_vecs = []
for tid in range(tokenizer.vocab_size):
    tok = tokenizer.convert_ids_to_tokens(tid)
    if tok in word2index:
        id2w2v_list.append(word2index[tok])
    elif tok == tokenizer.pad_token:
        id2w2v_list.append(orig_vocab_size + len(extra_vecs))
        extra_vecs.append(torch.zeros(embedding_dim, device=device))
    else:
        id2w2v_list.append(orig_vocab_size + len(extra_vecs))
        extra_vecs.append(torch.randn(embedding_dim, device=device) * 0.01)

if extra_vecs:
    extra_vecs = torch.stack(extra_vecs, dim=0)
    full_w2v = torch.cat([base_w2v, extra_vecs], dim=0)
else:
    full_w2v = base_w2v

id2w2v = torch.tensor(id2w2v_list, dtype=torch.long, device=device)

#Modules
class QueryMapping(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, 1024)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))


class SensoryT5Model(nn.Module):
    def __init__(self, num_labels, w2v_weight, id2w2v_map):
        super().__init__()
        self.t5 = T5Model.from_pretrained("t5-large")

        # I freeze the external embedding and only look it up by index
        self.w2v_table = nn.Embedding.from_pretrained(w2v_weight, freeze=True)
        self.register_buffer("id2w2v", id2w2v_map)

        self.query_mapping = QueryMapping()
        self.attn = nn.MultiheadAttention(embed_dim=1024, num_heads=16)
        self.classifier = nn.Linear(1024, num_labels)
        self.log_softmax = nn.LogSoftmax(dim=1)

    @staticmethod
    def mean_pool(hidden_states, mask):
        mask = mask.unsqueeze(-1).type_as(hidden_states)
        summed = (hidden_states * mask).sum(dim=1)
        count  = mask.sum(dim=1).clamp_min(1e-9)
        return summed / count

    def forward(self, input_ids, decoder_input_ids, labels, attention_mask=None):
        t5_out = self.t5(input_ids=input_ids,
                         decoder_input_ids=decoder_input_ids,
                         return_dict=True,
                         output_hidden_states=True)

        last_hidden = t5_out.decoder_hidden_states[-1]  # [b, s, 1024]
        t5_k = self.t5.decoder.block[-1].layer[0].SelfAttention.k(last_hidden)

        w2v_idx = self.id2w2v[input_ids]       
        sens_vec = self.w2v_table(w2v_idx)      

        q = self.query_mapping(sens_vec).transpose(0, 1)  
        k = t5_k.transpose(0, 1)                           
        v = last_hidden.transpose(0, 1)                   

        attn_out, _ = self.attn(q, k, v)                  
        attn_out = attn_out.transpose(0, 1)             

        if attention_mask is None:
            attention_mask = (input_ids != PAD_ID).long()

        fused = attn_out * last_hidden
        pooled = self.mean_pool(fused, attention_mask)

        logits = self.log_softmax(self.classifier(pooled))
        if labels is not None:
            loss = nn.NLLLoss()(logits, labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
