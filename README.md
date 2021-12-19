# Interactive Path Reasoning on Graph for Conversational Recommendation
Conversational Path Reasoning ([CPR](https://dl.acm.org/doi/pdf/10.1145/3394486.3403258)) framework  introduce graph to address the multi-round conversational recommendation problem. It
tackles what item to recommend and what attribute to ask problem
through message propagation on the graph.

If you have any question regarding the CPR System, please contact its main author, Mr. Gangyi Zhang: gangyi.zhang@outlook.com



Please kindly cite our paper if you use our code/dataset!
```
@inproceedings{lei2020interactive,
  title={Interactive Path Reasoning on Graph for Conversational Recommendation},
  author={Lei, Wenqiang and Zhang, Gangyi and He, Xiangnan and Miao, Yisong and Wang, Xiang and Chen, Liang and Chua, Tat-Seng},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={2073--2083},
  year={2020}
}
```

##### Shortcuts:

**Code:**
```
https://cpr-conv-rec.github.io/SCPR.zip
```
**Data:**
```
- Google Drive： https://drive.google.com/file/d/1NSVwwIPpWsbfvgvyT48ZUbVdkJBjM4M7/view?usp=sharing
- Tencent Weiyun：https://share.weiyun.com/LletCCdF
```

---

**This is our torch implementation for the paper:**
## Environment Requirement
* Python >= 3.6
* Numpy >= 1.12
* PyTorch >= 1.0

## Example to Run the Code
**1. Graph Construction**
```
python graph_init.py --data_name <data_name>
```
`<data_name>` is one of `{LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}`

###### Note: 
- `LAST_FM_STAR` and  `YELP_STAR` using the original attributes (pruning
off frequency &lt; 10 attributes) for binary question scenario. 
- Following the setting of [EAR](https://arxiv.org/abs/2002.09102), `LAST_FM` is designed to evaluate binary question scenario by merging relevant attributes into  coarse-grained attributes and `YELP` is designed for enumerated questions by builting a 2-layer taxonomy.

**2. Train FM Embedding**
```
python FM_train.py --data_name <data_name>
```
**More Details:**

Use `python FM_train.py -h` to get more argument setting details.

```
  -h, --help             show this help message and exit
  -lr <lr>              learning rate
  -flr <flr>            learning rate of feature similarity learning
  -bs <bs>              batch size
  -hs <hs>              hidden size & embedding size
  -dr <dr>              dropout ratio
  -uf <uf>              update feature
  -me <me>              the number of train epoch
  -seed <seed>          random seed
  --data_name <data_name>
                        One of {LAST_FM, LAST_FM_STAR, YELP, YELP_STAR}.

```


**3. Train RL Agent & Evaluate**
```
python RL_model.py --data_name <data_name> --fm_epoch <the epoch of FM embedding>
```
###### Note:
- The default `fm_epoch` is `0`, which means the FM embedding we trained in a particular FM epoch. To run quickly, you can use this preset FM embedding  for RL training, which can be found in the `tmp/<data_name>/FM-model-embeds`.

**More Details:**


Use `python RL_model.py -h` to get more argument setting details.



```
  -h, --help            show this help message and exit
  --seed <seed>         random seed.
  --epochs <epochs>     the number of RL train epoch.
  --fm_epoch <fm_epoch> the epoch of FM embedding
  --batch_size <batch_size>
                        batch size.
  --gamma <gamma>       reward discount factor.
  --lr <lr>             learning rate.
  --hidden <hidden>     hidden size
  --memory_size <memory_size>
                        the size of memory
  --data_name <data_name>
                        One of {LAST_FM*, LAST_FM, YELP*, YELP}.
  --entropy_method <entropy_method>
                        entropy_method is one of {entropy, weight entropy}
  --max_turn <max_turn>
                        max conversation turn
  --ask_num <attr_num>   the number of attributes for <data_set>
  --observe_num <observe_num>
                        the number of epochs to save RL model and metric
  --target_update <target_update>
                        the number of epochs to update policy parameters

```




## Dataset
We provide two processed datasets: Last-FM, Yelp.
* You can find the full version of recommendation datasets via [Last-FM]( https://grouplens.org/datasets/hetrec-2011/), [Yelp](https://www.yelp.com/dataset/)
* Here we list the relation types in different datasets to let readers to get
better understanding of the dataset.

<table>
  <tr>
    <th colspan="2">Dateset</th>
    <th>LastFM*</th>
    <th>Yelp*</th>
  </tr>
  <tr>
    <td rowspan="4">User-Item<br>Interaction</td>
    <td>#Users</td>
    <td>1,801</td>
    <td>27,675</td>
  </tr>
  <tr>
    <td>#Items</td>
    <td>7,432</td>
    <td>70,311</td>
  </tr>
  <tr>
    <td>#Interactions</td>
    <td>76,693</td>
    <td>1,368,606</td>
  </tr>
  <tr>
    <td>#attributes</td>
    <td>33</td>
    <td>29</td>
  </tr>
  <tr>
    <td rowspan="3">Graph</td>
    <td>#Entities</td>
    <td>9,266</td>
    <td>98,605</td>
  </tr>
  <tr>
    <td>#Relations</td>
    <td>4</td>
    <td>3</td>
  </tr>
  <tr>
    <td>#Triplets</td>
    <td>138,217</td>
    <td>2,884,567</td>
  </tr>
  <tr>
    <th>Relations</th>
    <th>Description</th>
    <th colspan="2">Number of Relations</th>
  </tr>
  <tr>
    <td>Interact</td>
    <td>user---item</td>
    <td>76,696</td>
    <td>1,368,606</td>
  </tr>
  <tr>
    <td>Friend</td>
    <td>user---user</td>
    <td>23,958</td>
    <td>688,209</td>
  </tr>
  <tr>
    <td>Like</td>
    <td>user---attribute</td>
    <td>7,276</td>
    <td>*</td>
  </tr>
  <tr>
    <td>Belong_to</td>
    <td>item---attribute</td>
    <td>30,290</td>
    <td>350,175</td>
  </tr>
</table>

---


<table>
  <tr>
    <th colspan="2">Dateset</th>
    <th>LastFM</th>
    <th>Yelp</th>
  </tr>
  <tr>
    <td rowspan="4">User-Item<br>Interaction</td>
    <td>#Users</td>
    <td>1,801</td>
    <td>27,675</td>
  </tr>
  <tr>
    <td>#Items</td>
    <td>7,432</td>
    <td>70,311</td>
  </tr>
  <tr>
    <td>#Interactions</td>
    <td>76,693</td>
    <td>1,368,606</td>
  </tr>
  <tr>
    <td>#attributes</td>
    <td>8,438</td>
    <td>590</td>
  </tr>
  <tr>
    <td rowspan="3">Graph</td>
    <td>#Entities</td>
    <td>17,671</td>
    <td>98,576</td>
  </tr>
  <tr>
    <td>#Relations</td>
    <td>4</td>
    <td>3</td>
  </tr>
  <tr>
    <td>#Triplets</td>
    <td>228,217</td>
    <td>2,533,827</td>
  </tr>
  <tr>
    <th>Relations</th>
    <th>Description</th>
    <th colspan="2">Number of Relations</th>
  </tr>
  <tr>
    <td>Interact</td>
    <td>user---item</td>
    <td>76,696</td>
    <td>1,368,606</td>
  </tr>
  <tr>
    <td>Friend</td>
    <td>user---user</td>
    <td>23,958</td>
    <td>688,209</td>
  </tr>
  <tr>
    <td>Like</td>
    <td>user---attribute</td>
    <td>33,120</td>
    <td>*</td>
  </tr>
  <tr>
    <td>Belong_to</td>
    <td>item---attribute</td>
    <td>94,446</td>
    <td>477,012</td>
  </tr>
</table>

---
## Data Description
**1. Graph Generate Data**

* `user_item.json`
  * Interaction file.
  * A dictionary of key value pairs. The key and the values of a dictionary entry: [`userID` : `a list of itemID`].
  
* `tag_map.json`
  * Map file.
  * A dictionary of key value pairs. The key and the value of a dictionary entry: [`Real attributeID` : `attributeID`].
  
* `user_dict.json`
  * User file.
  *  A dictionary of key value pairs. The key is `userID` and the value of a dictionary entry is a new dict: (''friends'' : `a list of userID`) & [''like'' : `attributeID`]
  
* `item_dict.json`
  * Item file.
  * A dictionary of key value pairs. The key is `itemID` and the value of a dictionary entry is a new dict: [''attribute_index'' : `a list of attributeID`] 

**2. FM Sample Data**
###### For the process of generating FM train data, please refer to Appendix B.2 of the paper.
* `sample_fm_data.pkl`
  *  The pickle file consists of five lists, and the fixed index of each list forms a training tuple`(user_id, item_id, neg_item, cand_neg_item, prefer_attributes)`.
           
        
            
```
user_pickle = pickle_file[0]           user id
item_p_pickle = pickle_file[1]         item id that has interacted with user
i_neg1_pickle = pickle_file[2]         negative item id that has not interacted with user
i_neg2_pickle = pickle_file[3]         negative item id that has not interacted with the user in the candidate item set
preference_pickle = pickle_file[4]     the user’s preferred attributes in the current turn
```

**3. UI Interaction Data**

* `review_dict.json`
    *  Items that the user has interacted with
    *  Used for generating FM sample data
    *  Used for training and testing in RL

    