# Interactive Path Reasoning on Graph for Conversational Recommendation
Conversational Path Reasoning (CPR) framework  introduce graph to address the multi-round conversational recommendation problem. It
tackles what item to recommend and what attribute to ask problem
through message propagation on the graph

This is our torch implementation for the paper:


## Environment Requirement
* Python >= 3.6
* Numpy >= 1.12
* PyTorch >= 1.0

## Example to Run the Code
**1. Graph Construction**
```
python Main.py --data_name <data_name>
```
''<data_name>'' is one of {''LAST_FM*'', ''LAST_FM'', ''YELP*'', ''YELP''}

Note: ''LAST_FM*'' and  ''YELP*'' using the original attributes (pruning
off frequency &lt; 10 attributes) for binary question scenario. Following the setting of [EAR](https://arxiv.org/abs/2002.09102), ''LAST_FM'' is designed to evaluate binary question scenario by merging relevant attributes into  coarse-grained attributes and ''YELP'' is designed for enumerated questions by builting a 2-layer taxonomy.

**2. Train FM Embedding**
```
python FM_train.py --data_name <data_name>
```
**More Details:**

Use `python FM_train.py -h` to get more argument setting details.

```
  -h, --help             show this help message and exit
  --lr <lr>              learning rate
  --flr <flr>            feature update Learning Rate
  --bs <bs>              batch size
  --hs <hs>              hidden size
  --dr <dr>              dropout ratio
  --uf <uf>              update feature
  --seed <seed>          random seed
  --data_name <data_name>
                        One of {LAST_FM*, LAST_FM, YELP*, YELP}.

```


**3. Train RL Agent & Evaluate**
```
python RL_model.py --data_name <data_name>
```
**More Details:**

Use `python RL_model.py -h` to get more argument setting details.

```
  -h, --help            show this help message and exit
  --seed <seed>         random seed.
  --epochs <epochs>     Max number of epochs.
  --fm_epoch <fm_epoch> the epoch of FM embedding
  --batch_size <batch_size>
                        batch size.
  --gamma <gamma>       reward discount factor.
  --lr <lr>             learning rate.
  --hidden <hidden>     hidden size
  --memory_size <memory_size>
                        size of memory
  --data_name <data_name>
                        One of {LAST_FM*, LAST_FM, YELP*, YELP}.
  --entropy_method <entropy_method>
                        entropy_method one of {entropy, weight entropy}
  --max_turn <max_turn>
                        max conversation turn
  --ask_num <ask_num>   the number of asking feature
  --observe_num <observe_num>
                        the number of epochs to save mode and metric
  --target_update <target_update>
                        epoch number: update policy parameters

```




## Dataset
We provide three processed datasets: Last-FM, Yelp.
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


* `user_item.json`
  * Interaction file.
  * A dictionary of key value pairs. The key and the values of a dictionary entry: [`userID` : `a list of itemID`].
  
* `tag_map.json`
  * Map file.
  * A dictionary of key value pairs. The key and the value of a dictionary entry: [`Real userID` : `userID`].
  
* `user_dict.json`
  * User file.
  *  A dictionary of key value pairs. The key is (`userID`) and the value of a dictionary entry is a new dict: (''friends'' : `a list of userID`) & [''like'' : `attributeID`)
 
  
* `item_dict.json`
  * Item file.
  * A dictionary of key value pairs. The key is (`itemID`) and the value of a dictionary entry is a new dict: (''attribute_index'' : `a list of attributeID`) & (''real_ID'' : `real_ID`)

  