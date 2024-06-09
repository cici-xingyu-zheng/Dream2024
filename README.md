Here we record some result metrics and TO-DOs. Let's go!

## RESULTS

---
We are currently using Sergey's updated Deepnose embeddings trained on the Leffingwell dataset (task is 3000 mono molecules for descriptor scoring, descriptor represented with Word2Vec?).

We use RF and RGBoost models to see if Deepnose features is predictable for perceptual distance. 

### 05/21/24

Parameter random search with feature combination method == "avg":
```
RandomForest:
R mean: 0.624009999222098
R std: 0.01012993384541656
RMSE mean: 0.12392184704104232
RMSE std: 0.0005575734739967653

XGBoost:
R mean: 0.5722108295854037
R std: 0.012305155699621435
RMSE mean: 0.13011032718717613
RMSE std: 0.0007585502287922674
```
### 05/22/24
Parameter random search with feature combination method == "log":
```
RandomForest:
R mean: 0.5827653094074099
R std: 0.013314527423509323
RMSE mean: 0.12818056230574132
RMSE std: 0.0010312395376071216

XGBoost:
R mean: 0.6179364843801805
R std: 0.0085436109653248
RMSE mean: 0.12423745074940677
RMSE std: 0.0007641590472822367
```
### 05/27/24
1. Parameter random search with feature combination method == "max":
```
RandomForest:
R mean: 0.5944725056504726
R std: 0.010848141551718406
RMSE mean: 0.12615760763131553
RMSE std: 0.0006135566872935029

XGBoost:
R mean: 0.6201911367008879
R std: 0.006041495836494987
RMSE mean: 0.12340290856833196
RMSE std: 0.0005129668557081593
```

2. **Add beta parameter**:

``` log_sum_exp_beta(x_1, x_2, ..., x_n) = (1/beta) * log(sum(exp(beta * x_i)))```

- When beta > 1, the operation becomes more "peaked", emphasizing the maximum value in the input. 

- When 0 < beta < 1, the operation becomes smoother, giving more weight to the smaller values in the input.

- As beta approaches 0, the log-sum-exp operation with the beta parameter approximates the arithmetic mean (linear combination) of the input values.

- As beta approaches infinity, the log-sum-exp operation with the beta parameter approximates the maximum value of the input values, emphasizing the most dominant component.

We did a beta sweep with beta = 0.1 - 10; bigger beta perform slightly better (See `Output/beta_sweep.txt`)

3. **Add dragon**:

We also tried on the Dragon descriptors provided by the organizers. In a standard initalization, mixture combined by averaging, although the code runs very slowly (Feature Dim ~9000, training > 30 min) we were able to obtain comparible results as Deenose features:

``` 
Random Forest - R: 0.595
Random Forest - RMSE: 0.126

XGBoost - R: 0.544
XGBoost - RMSE: 0.133
```

Blindly stacking deepnose and dragon doesn't seem to help too much:

```
Random Forest - R: 0.585
Random Forest - RMSE: 0.127

XGBoost - R: 0.551
XGBoost - RMSE: 0.132
```

4. Q&A: 
	- why we see Snitz 1 spread differently?
		- ''We prepared several different versions for each mixture size containing 1, 4, 10, 15, 20, 30, 40 or 43 components, such that half of the versions were **wellspread in perceptual space**, and half of the versions were **wellspread in physicochemical space**.''
	- Does 500 contains identical?
		- Nope;
	- What are the avaiable GCN strctures?
		- I located where it got mentioned in the Webnar: 00:27:44. Don't know what this team is though.
		- Michael Schmuker https://github.com/Huitzilo
	- The Dhurandhar paper code: https://github.com/jeriscience/OlfactionAD


### 05/31/24
**Planning:**
- for Deepnose features, try using difference:
	- For mean: by sum
	- For log: by division

- for Dragon features:
	- first, clean up the same value;
	- second, plot svd;
	- then:
		1. log normal distribution of features?
			- try different ways of normalizing;
		2. try reducing the dimension
- we need to figure out some confusion matrix like things, that allow us to know which dataset perform worse
	- Bushdid underperform consistently using different deepnose feature combinations
- how to include identical?

**Progress:**
1. We plotted the deepnose features distribution, and found out that it looks more like log normal; therefore we try out log the features first then standard transform; results stand out from random seed; In the optimization round, the result is slightly better and more robust:
```
RandomForest Average Performance:
R mean: 0.6258733379314186
R std: 0.007713393434087301
RMSE mean: 0.12375755904465387
RMSE std: 0.0005545934581237493

XGBoost Average Performance:
R mean: 0.6149325792301861
R std: 0.00786442684116577
RMSE mean: 0.125260023134671
RMSE std: 0.0006472561753324663
```

2. Distance features (TD;LR: doesn't work that well):

For log distance, with random seed:
```
Random Forest - R: 0.598
Random Forest - RMSE: 0.126

XGBoost - R: 0.561
XGBoost - RMSE: 0.131
```

For avg distance, with random seed:
```
Random Forest - R: 0.560
Random Forest - RMSE: 0.130

XGBoost - R: 0.552
XGBoost - RMSE: 0.132
```

For log standard, then avg distance, with random seed:
```
Random Forest - R: 0.588
Random Forest - RMSE: 0.127

XGBoost - R: 0.539
XGBoost - RMSE: 0.134
```

Okay stacked, with random seed, - there seems to be improvement:
```
Random Forest - R: 0.592
Random Forest - RMSE: 0.126

XGBoost - R: 0.538
XGBoost - RMSE: 0.134
```

We decided to stack the difference features and optimize over that. After optimization, the mean performance is not as imporessive as expected so we will perhaps still use the concatinated features for now; to consider combining with phsysicalchemical features.

3. Tried out Mordred Descriptors (TD;LR: usable feature space; even better when combined with Deepnose features) 

After fighting with the Dragon Descriptors that the Synapse website provided, that does not have enetries for all CIDs, see `Dragon_feagures.ipynb`, we decided to use Mordred, and have installed the module and produced the descriptpors ourselves. I've output the features as well, it's called `Mordred_features_in-house.csv`. 

With Random seed, and log standard transformed, and averaging to create mixtures, training takes about 10 min.

```
Random Forest - R: 0.606
Random Forest - RMSE: 0.125

XGBoost - R: 0.557
XGBoost - RMSE: 0.131
```

The mean abs. difference between prediction to true value:
```
Dataset
Bushdid    0.103358
Ravia      0.098455
Snitz 1    0.084759
Snitz 2    0.102151
```
Compared to using deepnose features, does better in Bushdid:
```
Dataset
Bushdid    0.122663
Ravia      0.098067
Snitz 1    0.064842
Snitz 2    0.073770
```

Stacked with Deepnose also log standard transformed, random seeds:

```
Random Forest - R: 0.622
Random Forest - RMSE: 0.123

XGBoost - R: 0.543
XGBoost - RMSE: 0.134
```

Absolute diff between pred and true:
```
Dataset
Bushdid    0.118434
Ravia      0.098279
Snitz 1    0.066827
Snitz 2    0.077048
```

Woohoo! For deepnose features, presumably their importance would be more evenly distributed.
We would like to see if we can drop some of Mordred features, and if augmentation might work to exploit those feature spaces.

4. Worked on a round of Feature reduction... successfully removed 178 Descriptors that consistently have feature importance < 0.0001; see `mordred_feature_reduction.py`.

5. Identical augmentation (TD;LR: haven't got it to work)

### 06/04/24
Since the signmoid has a very long lagging period before it goes to 1, we might need to scale and systematically scale (from 0.6 - 0.95)

Optimization results:

Best RF so far at scale = 6 (intensity assumed 0.6, in a 0-1 scale)
```
RandomForest Average Performance:
R mean: 0.6276453974001096
R std: 0.005341367310099413
RMSE mean: 0.12417947362024453
RMSE std: 0.0006251847416127161
```

Best XGB so far at scale = 6.5 (intensity assumed 0.65, in a 0-1 scale)
```
XGBoost Average Performance:
R mean: 0.6308365798346898
R std: 0.005496425576258286
RMSE mean: 0.12391892164379108
RMSE std: 0.0004695359298822691
```

From random seeds with the best RF hyperparameters:
```
Dataset
Bushdid    0.125092
Ravia      0.080492
Snitz 1    0.065308
Snitz 2    0.076263
```
It seems that intensity augmentation helped Ravia's deviation to be inproved; but not so much for the other datasets.

We will try scaling from (.15 - .55) just in case we miss from the lower end.

Seperate: 
```
Slope for Snitz 1: 1.489
Slope for Snitz 2: 1.665
Slope for Ravia: 1.607
Slope for Bushdid: 1.403
```
x: y_pred y: y_true

Intensity augmentation has been implemented; but let's plot things and print things so that we know no mistakes are made, and then go from there.

---

## TO-DO:

- think about the relationship between the value obtained from different paradigm
- try distance (1. difference both ways; 2. difference + average)
- Mordred versions community: https://github.com/JacksonBurns/mordred-community
- MLP or SVM regression uh
- Try Morgan once Cyrille shared with me; Morgan + Leffingwell?
- think about how to reduce dimensionality
- intensity aided
- try the Snitz normalization
- Plot std by bootstrapping

### 05/24/24 Discussion with Sergey:

1. explore beta (grid search);
	- done
2. compare/combine Mordred descriptor;
3. implement max pool;
	- added
4. try Pearson as the creterion for the sake of it;
	- don't think it make sense for decision trees..


### 05/31/24 Extended Datasets:

What I imagine the format of the spreadsheet can be:

```
# extended_training_set
'Dataset', 'Mixture 1', 'Mixture 2', 'Experimental Values', 'Experimental Type', 
'Exp1',     1,           2,          .33,                   'rate' # (or 'tri')
```
Column names, and an example row.
```
# extended_mxiture_IDs
'Dataset' 'Mixture Label'	'CID'	'CID.1'	'CID.2'	'CID.3'	'CID.4'	'CID.5'	'CID.6'	...
```
Column names.

```
# extended_molecule_intensites (if avaiable)
'Dataset' 'Mixture Label'	'CID'	'CID.1'	'CID.2'	'CID.3'	'CID.4'	'CID.5'	'CID.6'	...
```
Column names.

First it'd be great to have the `molecule_intensites` dataframe for the experiment 4 and 5 in Ravia 2020. I would imagine that it has the same columns as `extended_molecule_intensites` described above.

- If two concentrations are used, they can each be a row of data.

- **Discuss**: I am not sure how many experimental values are reported based on different `Experimental Type`, as the have their own rating task, they also did the triangular and the two-alternative same–different task — but we can figure this out together.

Second, to extend the data to other Ravia experiments, it would be lovely to have:

1. experiment 1
2. experiment 2
3. experiment 3, the perfume one (if the CIDs are available)
4. experiment 6



---
## IDEAS (good or bad)

1. Make use of multiple bigger dataset:
	1. Combine Deepnose trained on different descriptor sets, using alignment like "cross model fine-tuning aligned and refined"
		- but not sure if Deepnose does well on all of them
	
	2. Fine tune the descriptors with domain specific texts
		- might be really more involved than we thought
	
2. Improve Deepnose's design: I heard recently there are some ways to do 3D invariance learning for molecules (not GNNs)

3. Make use of other datasets (same mixture scores and non equal intensity mixtures (concentraion?)): need to read the papers first

4. Combine with Dragon features 


Just for the sake of discussion:

1. Diffusion models for molecule to learn the spatial distribution??? As there are so many odor molecules

2. Alphafold 3 ??