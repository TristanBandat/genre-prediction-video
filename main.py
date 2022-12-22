import pandas as pd
# import data.music4_all_onion_dc
import tensorflow_datasets as tfds


def main():
    ds = tfds.load('music4_all_onion_dc', data_dir='data/')['train']
    for elem in ds:
        print(elem['input'].numpy())
    # filename_incp = 'data/id_incp.tsv'
    # filename_labels = 'data/id_genres_binary.tsv'
    # incp_df = pd.read_csv(filename_incp, sep='\t')
    # labels_df = pd.read_csv(filename_labels, sep='\t')
    # for i, line in labels_df.iterrows():
    #     test = labels_df[labels_df['id'] == line['id']][1:]
    #     indices = line.to_numpy()[1:].nonzero()
    #     for index in indices:
    #         labels_df.at[i, labels_df.columns[index+1]] = 1.0
    # labels_df.to_csv('data/id_genres_binary.tsv', sep='\t')

    pass


if __name__ == '__main__':
    main()
