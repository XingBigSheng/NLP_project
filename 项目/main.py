import math
import torch
import torch.nn as nn
import torch.optim as optim
import give_valid_test


device = torch.device("cpu")


class CellLSTM(nn.Module):
    def __init__(self):
        super(CellLSTM, self).__init__()

        self.W_xi = torch.randn(emb_size, n_hidden)
        self.W_xf = torch.randn(emb_size, n_hidden)
        self.W_ig = torch.randn(emb_size, n_hidden)
        self.W_xo = torch.randn(emb_size, n_hidden)

        self.W_hi = torch.randn(n_hidden, n_hidden)
        self.W_hf = torch.randn(n_hidden, n_hidden)
        self.W_hg = torch.randn(n_hidden, n_hidden)
        self.W_ho = torch.randn(n_hidden, n_hidden)

        self.b_i = torch.randn(1, n_hidden)
        self.b_f = torch.randn(1, n_hidden)
        self.b_g = torch.randn(1, n_hidden)
        self.b_o = torch.randn(1, n_hidden)

        self.W_hq = torch.randn(n_hidden, emb_size)
        self.b_q = torch.randn(1, emb_size)

        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))

    def forward(self, inputs):
        batch_size = inputs.size()[1]  # 保证batch_size 正确
        # 下三行为初始化
        outputs = []
        H = torch.randn(batch_size, n_hidden)
        C = torch.randn(batch_size, n_hidden)
        for X in inputs:
            I = torch.sigmoid(torch.mm(X, self.W_xi) + torch.mm(H, self.W_hi) + self.b_i)
            F = torch.sigmoid(torch.mm(X, self.W_xf) + torch.mm(H, self.W_hf) + self.b_f)
            G = torch.tanh(torch.mm(X, self.W_ig) + torch.mm(H, self.W_hg) + self.b_g)
            O = torch.sigmoid(torch.mm(X, self.W_xo) + torch.mm(H, self.W_ho) + self.b_o)
            C = torch.mul(F, C) + torch.mul(I, G)
            H = torch.mul(O, torch.tanh(C))
            outputs.append(H)
        return torch.stack(outputs, 0)


class TextLSTM(nn.Module):
    def __init__(self, n_layers):
        super(TextLSTM, self).__init__()
        self.n_layers = n_layers

        self.C = nn.Embedding(n_class, embedding_dim=emb_size)
        self.W = nn.Linear(n_hidden, n_class, bias=False)
        self.b = nn.Parameter(torch.ones([n_class]))
        # 声明LSTM层
        self.LSTM_ls = [CellLSTM() for i in range(n_layers)]
        # LSTM层中间的过度线性层
        self.W_n = torch.randn(n_hidden, emb_size)
        self.b_n = torch.randn(1, emb_size)

    def forward(self, inputs):
        inputs = self.C(inputs)
        inputs = inputs.transpose(0, 1)  # X : [n_step, batch_size, n_class]
        temoutputs = inputs
        for i in range(self.n_layers - 1):
            temoutputs = torch.matmul(self.LSTM_ls[i](temoutputs), self.W_n) + self.b_n
        outputs = self.LSTM_ls[self.n_layers - 1](temoutputs)
        outputs = outputs[-1]
        model_output = self.W(outputs) + self.b
        return model_output


def make_batch(train_path, word2number_dict, batch_size, n_step):
    all_input_batch = []
    all_target_batch = []

    text = open(train_path, 'r', encoding='utf-8')  # open the file

    input_batch = []
    target_batch = []
    for sen in text:
        word = sen.strip().split(" ")  # space tokenizer

        if len(word) <= n_step:  # pad the sentence
            word = ["<pad>"] * (n_step + 1 - len(word)) + word

        for word_index in range(len(word) - n_step):
            input = [word2number_dict[n] for n in word[word_index:word_index + n_step]]  # create (1~n-1) as input
            target = word2number_dict[
                word[word_index + n_step]]  # create (n) as target, We usually call this 'casual language model'
            input_batch.append(input)
            target_batch.append(target)

            if len(input_batch) == batch_size:
                all_input_batch.append(input_batch)
                all_target_batch.append(target_batch)
                input_batch = []
                target_batch = []

    return all_input_batch, all_target_batch


def make_dict(train_path):
    text = open(train_path, 'r', encoding='utf-8')  # open the train file
    word_list = set()  # a set for making dict

    for line in text:
        line = line.strip().split(" ")
        word_list = word_list.union(set(line))

    word_list = list(sorted(word_list))  # set to list

    word2number_dict = {w: i + 2 for i, w in enumerate(word_list)}
    number2word_dict = {i + 2: w for i, w in enumerate(word_list)}

    # add the <pad> and <unk_word>
    word2number_dict["<pad>"] = 0
    number2word_dict[0] = "<pad>"
    word2number_dict["<unk_word>"] = 1
    number2word_dict[1] = "<unk_word>"

    return word2number_dict, number2word_dict


def train_rnnlm():
    model = TextLSTM(2)
    model.to(device)
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Training
    batch_number = len(all_input_batch)
    for epoch in range(all_epoch):
        count_batch = 0
        for input_batch, target_batch in zip(all_input_batch, all_target_batch):
            optimizer.zero_grad()

            # input_batch : [batch_size, n_step, n_class]
            output = model(input_batch)

            # output : [batch_size, n_class], target_batch : [batch_size] (LongTensor, not one-hot)
            loss = criterion(output, target_batch)
            ppl = math.exp(loss.item())
            if (count_batch + 1) % 50 == 0:
                print('Epoch:', '%04d' % (epoch + 1), 'Batch:', '%02d' % (count_batch + 1), f'/{batch_number}',
                      'lost =', '{:.6f}'.format(loss), 'ppl =', '{:.6}'.format(ppl))

            loss.backward()
            optimizer.step()

            count_batch += 1

        # valid after training one epoch
        all_valid_batch, all_valid_target = give_valid_test.give_valid(word2number_dict, n_step)
        all_valid_batch.to(device)
        all_valid_target.to(device)

        total_valid = len(all_valid_target) * 128
        with torch.no_grad():
            total_loss = 0
            count_loss = 0
            for valid_batch, valid_target in zip(all_valid_batch, all_valid_target):
                valid_output = model(valid_batch)
                valid_loss = criterion(valid_output, valid_target)
                total_loss += valid_loss.item()
                count_loss += 1

            print(f'Valid {total_valid} samples after epoch:', '%04d' % (epoch + 1), 'lost =',
                  '{:.6f}'.format(total_loss / count_loss),
                  'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))

        if (epoch + 1) % save_checkpoint_epoch == 0:
            torch.save(model, f'models/rnnlm_model_epoch{epoch + 1}.ckpt')


def test_rnnlm(select_model_path):
    model = torch.load(select_model_path, map_location="cpu")  # load the selected model

    # load the test data
    all_test_batch, all_test_target = give_valid_test.give_test(word2number_dict, n_step)
    total_test = len(all_test_target) * 128
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0
    count_loss = 0
    for test_batch, test_target in zip(all_test_batch, all_test_target):
        test_output = model(test_batch)
        test_loss = criterion(test_output, test_target)
        total_loss += test_loss.item()
        count_loss += 1

    print(f"Test {total_test} samples with {select_model_path}……………………")
    print('lost =', '{:.6f}'.format(total_loss / count_loss),
          'ppl =', '{:.6}'.format(math.exp(total_loss / count_loss)))


if __name__ == '__main__':
    n_step = 5  # number of cells(= number of Step)
    n_hidden = 5  # number of hidden units in one cell
    batch_size = 512  # batch size
    learn_rate = 0.001
    all_epoch = 200  # the all epoch for training
    emb_size = 128  # embeding size
    save_checkpoint_epoch = 100  # save a checkpoint per save_checkpoint_epoch epochs
    train_path = 'data/train.txt'  # the path of train dataset

    word2number_dict, number2word_dict = make_dict(train_path)  # use the make_dict function to make the dict
    print("The size of the dictionary is:", len(word2number_dict))

    n_class = len(word2number_dict)  # n_class (= dict size)

    all_input_batch, all_target_batch = make_batch(train_path, word2number_dict, batch_size, n_step)  # make the batch
    print("The number of the train batch is:", len(all_input_batch))

    all_input_batch = torch.LongTensor(all_input_batch).to(device)  # list to tensor
    all_target_batch = torch.LongTensor(all_target_batch).to(device)

    print("\nTrain the RNNLM……………………")
    train_rnnlm()

    # print("\nTest the RNNLM……………………")
    # select_model_path = "models/rnnlm_model_epoch2.ckpt"
    # test_rnnlm(select_model_path)
