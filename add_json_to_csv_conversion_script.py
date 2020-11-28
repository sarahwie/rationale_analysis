import json
import csv
import os
import sys

if __name__ == '__main__':

	infile = sys.argv[0]
	outfile = os.path.join(os.path.split(infile)[0], os.path.splitext(os.path.basename(infile))[0] + '.csv')

	print(infile)
	docs = []
	for line in open(infile, 'r'):
		docs.append(json.loads(line))   

	with open(outfile, 'w') as g:
		writer = csv.DictWriter(g, fieldnames=['item', 'text', 'pred_label', 'label'])
		writer.writeheader()
		for i, item in enumerate(docs):
			# convert to dictionary & write to csv file
			writer.writerow({'text':item['rationale']['document'], 'item':i, 'pred_label':item['metadata']['predicted_label'], 'label':item['metadata']['label']})

