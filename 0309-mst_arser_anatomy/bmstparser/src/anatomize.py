import dynet as dy
from utils import read_conll, write_conll
from operator import itemgetter
import utils, time, random, decoder, gzip
import numpy as np
import codecs
from linalg import *
from optparse import OptionParser
import utils
from mstlstm import MSTParserLSTM
import sys, os.path, time
import cPickle as pkl
import pdb


def rnn_mlp(self, sens):
	'''
	Here, I assumed all sens have the same length.
	'''
	words, pwords, pos, chars = sens[0], sens[1], sens[2], sens[5]
	# words: indices of words in wlookup.
	# words shape: sent_length x batch_size (length x batch)
	if self.options.use_char:
		cembed = [dy.lookup_batch(self.clookup, c) for c in chars]
		char_fwd, char_bckd = self.char_lstm.builder_layers[0][0].initial_state().transduce(cembed)[-1],\
							  self.char_lstm.builder_layers[0][1].initial_state().transduce(reversed(cembed))[-1]
		crnn = dy.reshape(dy.concatenate_cols([char_fwd, char_bckd]), (self.options.we, words.shape[0]*words.shape[1]))
		cnn_reps = [list() for _ in range(len(words))]
		for i in range(words.shape[0]):
			cnn_reps[i] = dy.pick_batch(crnn, [i * words.shape[1] + j for j in range(words.shape[1])], 1)

		wembed = [dy.lookup_batch(self.wlookup, words[i]) + dy.lookup_batch(self.elookup, pwords[i]) + cnn_reps[i] for i in range(len(words))]
	else:
		wembed = [dy.lookup_batch(self.wlookup, words[i]) + dy.lookup_batch(self.elookup, pwords[i]) for i in range(len(words))]
	posembed = [dy.lookup_batch(self.plookup, pos[i]) for i in range(len(pos))] if self.options.use_pos else None

	inputs = [dy.concatenate([w, pos]) for w, pos in zip(wembed, posembed)] if self.options.use_pos else wembed

	h_out = self.bi_rnn(inputs, words.shape[1], 0, 0) #self.deep_lstms.transduce(inputs)
	# h_out: python list of concatenated BiLSTM hidden state

	# BiLSTM hidden tape (python list --> dynet tensor)
	h = dy.concatenate_cols(h_out) # shape: batch x ( 2*rnn x len )

	# arc-head
	H = self.activation(dy.affine_transform([self.arc_mlp_head_b.expr(), self.arc_mlp_head.expr(), h]))
	# arc-modifier
	M = self.activation(dy.affine_transform([self.arc_mlp_dep_b.expr(), self.arc_mlp_dep.expr(), h]))
	# arc-head for label
	HL = self.activation(dy.affine_transform([self.label_mlp_head_b.expr(), self.label_mlp_head.expr(), h]))
	# arc-modifier for label
	ML = self.activation(dy.affine_transform([self.label_mlp_dep_b.expr(), self.label_mlp_dep.expr(), h]))

	return h, H, M, HL, ML


def bilinear(self, M, W, H, input_size, seq_len, batch_size, num_outputs=1, bias_x=False, bias_y=False):
	if bias_x:
		M = dy.concatenate([M, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])
	if bias_y:
		H = dy.concatenate([H, dy.inputTensor(np.ones((1, seq_len), dtype=np.float32))])

	nx, ny = input_size + bias_x, input_size + bias_y

	# Modifier * Weight * Head in two steps:

	# Modifier * Weight transposed
	left_lin = W * M
	if num_outputs > 1:
		left_lin = dy.reshape(left_lin, (ny, num_outputs * seq_len), batch_size=batch_size)

	# Weight * Head
	if num_outputs == 1:
		right_lin = dy.transpose(W) * H
	else:
		right_lin = W * H
	if num_outputs > 1:
		right_lin = dy.reshape(left_lin, (nx, num_outputs * seq_len), batch_size=batch_size)

	# blin = dy.transpose(H) * left_lin
	# if num_outputs > 1:
	# 	blin = dy.reshape(blin, (seq_len, num_outputs, seq_len), batch_size=batch_size)
	return left_lin, right_lin


def build_graph(self, mini_batch):
	seq_len = mini_batch[0].shape[0]
	batch_size = mini_batch[0].shape[1]

	# network: forward
	h, H, M, HL, ML = self.rnn_mlp(mini_batch)

	arc_left_coeff, arc_right_coeff = self.bilinear(M, self.w_arc.expr(), H, self.options.arc_mlp, mini_batch[0].shape[0], mini_batch[0].shape[1],1, True, False)

	rel_left_coeff, rel_right_coeff = self.bilinear(ML, self.u_label.expr(), HL, self.options.label_mlp, mini_batch[0].shape[0], mini_batch[0].shape[1], len(self.irels), True, True)

	outputs, dims = [h, H, M, HL, ML, arc_left_coeff, arc_right_coeff, rel_left_coeff, rel_right_coeff], [seq_len, batch_size]
	# dy.renew_cg()

	return outputs, dims


def probe(parser, buckets, probe_idx):
	pr_h = []
	pr_H = []
	pr_M = []
	pr_HL = []
	pr_ML = []
	pr_lcoeff = []
	pr_rcoeff = []
	pr_lcoeff_rel = []
	pr_rcoeff_rel = []

	for mini_batch in utils.get_batches(buckets, parser, False):
		outputs, dims = parser.build_graph(mini_batch)
		seq_len, batch_size = dims

		h, H, M, HL, ML, arc_left_coeff, arc_right_coeff, rel_left_coeff, rel_right_coeff = \
			map(lambda t: t.npvalue(), outputs)

		for i in range(batch_size):
			pr_h.append(h[:,probe_idx,i])
			pr_H.append(H[:,probe_idx,i])
			pr_M.append(M[:,probe_idx,i])
			pr_HL.append(HL[:,probe_idx,i])
			pr_ML.append(ML[:,probe_idx,i])
			pr_lcoeff.append(arc_left_coeff[:,probe_idx,i])
			pr_rcoeff.append(arc_right_coeff[:,probe_idx,i])
			pr_lcoeff_rel.append(rel_left_coeff[:,probe_idx,i])
			pr_rcoeff_rel.append(rel_right_coeff[:,probe_idx,i])

		dy.renew_cg()

	return [pr_h, pr_H, pr_M, pr_HL, pr_ML, pr_lcoeff, pr_rcoeff, pr_lcoeff_rel, pr_rcoeff_rel]


def write_probes(repl_words, probe_result, output_dir='../probe/1/'):
	fh, fH, fM, fHL, fML, f_lcoeff, f_rcoeff, f_lcoeff_rel, f_rcoeff_rel = \
		map(lambda s: open(os.path.join(output_dir,s), 'w+'), ['h', 'H', 'M', 'HL', 'ML', 'lcoeff', 'rcoeff', 'lcoeff_rel', 'rcoeff_rel'])
	for word, pr_h, pr_H, pr_M, pr_HL, pr_ML, pr_lcoeff, pr_rcoeff, pr_lcoeff_rel, pr_rcoeff_rel in zip(repl_words, *probe_result):
		map(lambda f, t: f.write( '\t'.join([word] + map(str, t.tolist())) + '\n' ),
			[fh, fH, fM, fHL, fML, f_lcoeff, f_rcoeff, f_lcoeff_rel, f_rcoeff_rel],
			[pr_h, pr_H, pr_M, pr_HL, pr_ML, pr_lcoeff, pr_rcoeff, pr_lcoeff_rel, pr_rcoeff_rel])


def main():
	MSTParserLSTM.rnn_mlp = rnn_mlp
	MSTParserLSTM.bilinear = bilinear
	MSTParserLSTM.build_graph = build_graph

	parser = OptionParser()
	parser.add_option("--extrn", dest="external_embedding", help="External embeddings", metavar="FILE", default="../data/sskip.100.vectors.gz")
	parser.add_option("--params", dest="params", help="Parameters file", metavar="FILE", default="../models/params.pickle")
	parser.add_option("--model", dest="model", help="Load/Save model file", metavar="FILE", default="../models/model-135")
	parser.add_option("--test", dest="conll_test", help="Annotated CONLL test file", metavar="FILE", default=None)

	parser.add_option("--repl-words", dest="repl_words", help="Words at probing position", metavar="FILE", default="../probe/repl.words")
	parser.add_option("--output-dir", dest="output_dir", help="Output directory", default="../probe/1")
	parser.add_option("--probe-index", type="int", dest="probe_idx", default=0)

	(options, args) = parser.parse_args()

	with open(options.params, 'r') as paramsfp:
		w2i, pos, rels, chars, stored_opt = pkl.load(paramsfp)
	stored_opt.external_embedding = options.external_embedding
	mstParser = MSTParserLSTM(pos, rels, w2i, chars, stored_opt)
	mstParser.Load(options.model)

	probe_buckets = [list()]
	probe_data = list(utils.read_conll(open(options.conll_test, 'r')))
	for d in probe_data:
		probe_buckets[0].append(d)

	probe_result = probe(mstParser, probe_buckets, options.probe_idx)

	repl_words = [word.strip() for word in open(options.repl_words,'r').readlines() if word.strip() != '']
	write_probes(repl_words, probe_result, options.output_dir)


if __name__ == '__main__':
	main()