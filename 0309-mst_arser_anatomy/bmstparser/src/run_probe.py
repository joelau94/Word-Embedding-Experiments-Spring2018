import os

test_files = ['../sentence_files/sent_file_{}'.format(i) for i in range(1, 15)]
output_dirs = ['../probe/{}'.format(i) for i in range(1, 15)]
probe_indices = [5, 3, 8, 26, 15, 5, 4, 15, 10, 6, 8, 22, 29, 18]
repl_words_files = ['../probe/repl.words' for i in range(1, 29)]

for test, out, idx, repl in zip(test_files, output_dirs, probe_indices, repl_words_files):
	if not os.path.exists(out):
		os.mkdir(out)
	os.system('python anatomize.py --test {} --output-dir {} --probe-index {} --repl-words {}'.format(
		test, out, idx, repl))