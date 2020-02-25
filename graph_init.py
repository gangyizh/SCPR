
import argparse
from utils import *
from Graph_generate.lastfm_data_process import LastFmDataset
from Graph_generate.lastfm_small_data_process import LastFmSmallDataset
from Graph_generate.lastfm_graph import LastFmGraph
from Graph_generate.yelp_data_process import YelpDataset
from Graph_generate.yelp_graph import YelpGraph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=YELP, help='One of {LAST_FM, LAST_FM_SMALL, YELP, YELP_SMALL}.')
    args = parser.parse_args()
    DatasetDict = {
        LAST_FM: LastFmDataset,
        LAST_FM_SMALL: LastFmSmallDataset,
        YELP: YelpDataset,
        YELP_SMALL: YelpDataset
    }
    GraphDict = {
        LAST_FM: LastFmGraph,
        LAST_FM_SMALL: LastFmGraph,
        YELP: YelpGraph,
        YELP_SMALL: YelpGraph
    }

    # Create 'data_name' instance for data_name.
    print('Load', args.data_name, 'data_name from file...')
    print(TMP_DIR[args.data_name])
    if not os.path.isdir(TMP_DIR[args.data_name]):
        os.makedirs(TMP_DIR[args.data_name])
    dataset = DatasetDict[args.data_name](DATA_DIR[args.data_name])
    save_dataset(args.data_name, dataset)
    print('Save', args.data_name, 'dataset successfully!')

    # Generate graph instance for data_name
    print('Create', args.data_name, 'graph from data_name...')
    dataset = load_dataset(args.data_name)
    kg = GraphDict[args.data_name](dataset)
    save_kg(args.data_name, kg)
    print('Save', args.data_name, 'graph successfully!')

if __name__ == '__main__':
    main()

