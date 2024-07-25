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

1. Intensity incoporation: 

	- **brief explanation again**: Ravia's data's not intensity matched; but they provide an intensity score for each molecule in the mixture, 
	and how to combine them, i.e. sum over the molecule with a weighing function based on the intensity:

```
weight = 1 / (1 + np.exp(-(x - alpha_int)/beta_int))
```

	- we used this formula to weigh for the Ravia data, and for Bushdid and Snitz, we optimize via varying **a fixed intensity weight** for all molecules.

Since the signmoid has a very long lagging period before it goes to 1, we tried from 0.1 to 0.9. 

Optimization results:

Best RF so far at intensity assumed around 0.6 and 0.2, in a 0-1 scale:

``` 
### intensity = 0.2:
RandomForest Average Performance:
R mean: 0.6375006086735041
R std: 0.006451750966573196
RMSE mean: 0.12165413362153225
RMSE std: 0.00047790539465141906

### intensity = 0.6:
RandomForest Average Performance:
R mean: 0.6276453974001096
R std: 0.005341367310099413
RMSE mean: 0.12417947362024453
RMSE std: 0.0006251847416127161
```


Best XGB so far at scale = 6.5 (intensity assumed 0.65, in a 0-1 scale)
```
### intensity = 0.25:
XGBoost Average Performance:
R mean: 0.6266051980823658
R std: 0.0067376094337279245
RMSE mean: 0.12429576133057188
RMSE std: 0.0007501798484626942

### intensity = 0.65:
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

Interestingly, when intensity = .2, vectors are very small in the overall scale in Bushdid and Snitz compared to Ravia, but maybe we need to plot and look at them seperately. 

2. Seperately, the predicted value v.s. true always has a bigger slope than 1, e.g.: 
```
### x: y_pred y: y_true
Slope for Snitz 1: 1.489
Slope for Snitz 2: 1.665
Slope for Ravia: 1.607
Slope for Bushdid: 1.403
```



3. Intensity augmentation has been implemented; but haven't plot things and print things so that we know no mistakes are made, and then go from there. 
But the overall impression is that it doesn't perform as well but not bad either. We haven't optimized anything yet.


Some conclusion is that without augmentation, the performance including intensity has been the best so far (RF; R =.6375), but without augmentating with other Ravia dataset; but we don't know if it helps in the actual testing data yet.

My thoughts is that there are two seperate things we can try as another attempt to combine Deepnose and physical chemical:

- First, find a relatively low Mordred dimension that is more or less on par with Deepnose;
- Second, combine Morgan fingerprint with leffingwell: which are both sparse data.

### 06/15/24

1. we ran an MLP and a SVR; to do a quick comparision
2. we subsampled the potential augmentation based on whether samples share mono molecules with the original training set; the average performance is not better; but more estimators (> 200 might help?) as we have more samples.

``` 
Random Forest Average Performance:
R mean: 0.6343168195209042
R std: 0.005635839804085785
RMSE mean: 0.12198357360915224
RMSE std: 0.0004758279324259476

XGBoost Average Performance:
R mean: 0.5835291550334958
R std: 0.016703216536849495
RMSE mean: 0.12677969614763263
RMSE std: 0.002275026499832646
```

``` 
# increase the serach for # of estimators
Random Forest Average Performance:
R mean: 0.6365621082108206
R std: 0.005377960512535479
RMSE mean: 0.12176980392680141
RMSE std: 0.0006470437287955823

XGBoost Average Performance:
R mean: 0.5835179973193518
R std: 0.01683486714022818
RMSE mean: 0.12679213018645616
RMSE std: 0.002315536795866572
```

### 06/18/24
The most recent idea we carried out was to have a lower-D physical-chemical feature set, that is relative same size scale as deepnose, instead of 10 times of it. First we thought SVD would do, but I very quickly realize that SVD low rank will give us the same dim back but just with a lower rank. So we decided to do a PCA on the covariance matrix of `X` being 161 x 1282 (# molecule, # features), `C = (1 / (n-1)) * (X^T * X)`, and do PCA on it (C is square matrix of dim. # of features). Say we select the top `k` principal components of the evectors, which has dim (1282 x k), and call it the projection matrix `P`. and then we get back the reduced feature matrix `X_reduced = X.matmul(P)`. We tried it on k = 96, being the same dim as Deepnose features, and the variance captured is about 100% of the original features.

Below are some comparison of performance:


1. Reduced Mordred alone: 
```
# Random seed:
Random Forest - R: 0.595
Random Forest - RMSE: 0.126
```

2. Deepnose with not-reduced Mordred:
```
# Random seed:
Random Forest - R: 0.622
Random Forest - RMSE: 0.123

# Optimized:
RandomForest Average Performance:
R mean: 0.6269922739474505
R std: 0.009665706205459547
RMSE mean: 0.12240838858892829
RMSE std: 0.0012340442648023965
```

3. Deepnose with reduced Mordred:
```
# Random seed:
Random Forest - R: 0.617
Random Forest - RMSE: 0.124

# Optimized:
RandomForest Average Performance:
R mean: 0.6312683272761064
R std: 0.008293668435681693
RMSE mean: 0.12391256611746541
RMSE std: 0.0004982968418171625

XGBoost Average Performance: # XGB also performs pretty well!
R mean: 0.6317021124916241
R std: 0.005763772259870198
RMSE mean: 0.1223166036600456
RMSE std: 0.00044259352066136496
```

4. Scale the stacked features with intensity; with distance, cosyne sim, and angle calculated seperately for deepnose and reduced dim mordred:

```
Best Random Forest model:
Hyperparameters: {'n_estimators': 400, 'min_samples_split': 2, 'max_features': 'auto', 'max_depth': 20, 'bootstrap': True}

RandomForest Average Performance:
R mean: 0.6392017624024392
R std: 0.006384713497650123
RMSE mean: 0.12066172317965651
RMSE std: 0.0008327914020346979

Best XGBoost model:
Hyperparameters: {'subsample': 0.7, 'n_estimators': 500, 'max_depth': 7, 'learning_rate': 0.01, 'colsample_bytree': 0.5}

XGBoost Average Performance:
R mean: 0.6416206117007875
R std: 0.004993599179229569
RMSE mean: 0.12002062605743546
RMSE std: 0.0004988276960423697
```

This is exciting as it is with intensity and feature combo we beat our previous performaces. We can also explore different lower D. 

We uploaded the features of Mordred, original (without imputation for NaN yet, but in ourprojection we used mean) and the reduced dim. Mordred to dropbox.

We will create a table now to record the inching up.

### 06/23/24

1. we will do the with scaling and without scaling; and a code that test the best param combo's per data mean RSME and R. 
2. we will write a first code that uses sparse input
3. forgot to do what Alex said about SVD projection.

### 06/30/24

1. Try out the selected features Cyrille has produced (random seed, not optimized)

| Features               |    R     | RMSE  |  Feature Dim |
| :--------------------- | :------  | :---- | :----------  |
| Mordred                |  0.599   | 0.126 | 70           |
| Dragon                 |  0.573   | 0.128 | 31           |
| Mordred normalized     |  0.603   | 0.125 | 77           |
| Dragon normalized      |  0.575   | 0.128 | 296          |
| Mordred + Deepnose     |  0.608   | 0.125 | 70 + 96      |
| Dragon + Deepnose      |  0.620   | 0.123 | 31 + 96      |
| Mordred normalized + DN|  0.617   | 0.124 | 77 + 96      |
| Dragon normalized + DN |  0.617   | 0.124 | 296 + 96     |

Note that for all selected features, we also post process normalize it. Feature dimension is before engineering. 
From the results, it seems that 
-  Mordred itself contains more information than Dragon, but Dragon when combined with Deepnose works pretty well too.
-  Stacking with Deepnose features helps, the best performance is on par with feature projection on a rough look.

We furthur try out `Dragon + Modred normalized + DN` just as this has out performed, we got `R = 0.614` and `RMSE = 0.124`, so overall okay. 

Optimization Results: Dragon/Mordred/Mordred normalized + Deepnose have on average > 0.63 performance; see `performace_summary.xlsx` for details.

2. Fix the CID 
	- Done; the correction happens for mixing, so does not affect too much the rest of the code;

3. Try out leaderboard, the procedure we took was to just use 10 model prediction's average:
	- First attempt is on Model 9; 
		- Use KNN imputation for `Dataset` feature 
		- What are the imputed values?

```
Random Forest - R: 0.712
Random Forest - RMSE: 0.121
```
Realize that one-hot has been encoded wrongly. Fixed now; will fix for new code.

### 07/05/24:
1. Realized that we never optimize the same way for Deepnose + unreduced Mordred. Doing that..
2. Is there a way actually, to diversify XGB performance as it is gradient based?
3. Finish wrapping the testing functions; haven't started leaderboard testing yet.

- found out the mistake in selected features; corrected.

4. Run in-house prepared Leaderboard testing for selected features; done!
5. Optimize for Dragon + Mordred + Deepnose; done; not beating the current best
6. For next week: varying the # of models (easy, but does not seem to change the result to much; tried 5, 10, 20); train different best models (the easier way is to expand on the random search program)

### 07/17/24:
1. We tried out bootstrapping, that for each model used in ensemble, we randomly bootstrap 500 samples -- doesn't help with performance when we tested it; (on Model 11, ends up with smaller R and bigger RMSE)
2. It seems that our random search isn't giving us the best generalizable performance; end of the day I might look into it
	- Nope it is actually pretty good; but I can imagine there might be way to get 5 differnet good-ish hyperparams, and average over them? Not sure if that will help...
3. Whether using 10 RF models equals to get 10x the set number of trees? 
	- Nope, with `bootstrap` set to be true, it's basically the same.
4. Whether our models fit on the training data with a different performance? So that a weighted sum can be trained on them?
	- Some observation, and potential explaination:

	1. Higher correlation for RF:
   		- RF's smoother predictions might be following the overall trend of the data more closely, leading to higher correlation.
   		- The averaging nature of RF might be capturing the general relationship well, even if individual predictions are slightly off.

	2. Lower MSE for XGB:
   		- XGB might be better at capturing specific patterns that reduce overall error, even if it doesn't always follow the trend as smoothly.
   		- XGB's ability to focus on harder-to-predict instances through boosting could be reducing the larger errors more effectively.

	- maybe we can use some weighing or stacking them to a linear regression.

5. Dependency issue solved.

### 07/22/24:
1. try out the oversampled Deepnose features; doesn't seem to outperform the previous one, at least combining with selected features;
2. try out combining predictions of the same feature set (our best so far); doing good but yet need to figure out how to use on the entire dataset with ensembles:

```
RF RMSE: 0.12197581633078773 
Correlation: 0.6474515913228636

XGB RMSE: 0.12131653217126667 
Correlation: 0.633644940861654

Combined RMSE: 0.1193005272566889 
Correlation: 0.6552910046267708

Average optimal thresholds: [0.40566357 0.61836108]
```
With that used on leaderboard:

``` 
Random Forest - R: 0.724
Random Forest - RMSE: 0.120

XGBoost - R: 0.716
XGBoost - RMSE: 0.117

Combined - R: 0.720
Combined - RMSE: 0.117
```

Will continue to try out different ways of mixing; but it does not fundamentally improve actually.

3. yet haven't combined model trained on different features. 
	- for the sparse features, we can potentially drop by importance, and add them to dense? or use them in a sequential model? Tried that with a threshold 0.001, was only able to shirnk Leffingwell.

I observe that when seperating Leffingwell and fingerprint they each do pretty well. I am running individual optimization now.

If the result looks good, I might want to optimize each of them respectively, for meta model training. 
	- Will try out Morgan alone, which might or might not generalize? (07/22/24) (took a peak doesn't look too good)
	- Can try on Leffingwell alone too? (did a basic trimming to 96 dim)
	- Can perhaps optimize combined sparse again...(done; but result looks not as good as before; how come?!)
	- Try Mordred alone in your train test and see if there's a bug (done; nope does not seem so)

Okay here is the plan. I still want to try 2 more ways of combining across features.

1) **(parallel) Meta model.** Train different model outputs y's with a regularized linear regression, so that the final prediction is a weighted average of multiple models. Basically, base level models make predictions independently; and the meta-model is trained to combine them.

	- first attempt is to combine sparse features and dense features that each outperforms. Well at least the results says that this procedure helps with RMSE but perhaps less with correlation, if the correlation of the original model is bad; this is a bit of a comfort that we perhaps don't need to try out many combos of not so good performance.
	- different meta model: I tried linear-ridge, polynomial-ridge, KNN, RF, and Gradient Boost. 


``` 
# with linear ridge:
Dense Model Performance:
  Correlation: 0.724
  RMSE: 0.120

Sparse Model Performance:
  Correlation: 0.683
  RMSE: 0.120

Meta Model Performance:
  Correlation: 0.714
  RMSE: 0.116
```

Results all:

```
Ridge - RMSE: 0.1152, Correlation: 0.7156
Poly_Ridge - RMSE: 0.1168, Correlation: 0.7099
RF - RMSE: 0.1139, Correlation: 0.7270
GB - RMSE: 0.1138, Correlation: 0.7250
KNN - RMSE: 0.1163, Correlation: 0.7081
```

So the currently winning one is a RF on two RFs.. `RF - RMSE: 0.1139, Correlation: 0.7270`. Although we haven't spent effort in optimizing the meta model.

2) **(sequential) Boosting on residuals.** Train one or few sparse feature model on the residuals of the dense feature model. In the current attempt we use the same dense and sparse feature set as the meta model. We used the hyper parameter for the best dense, and optimize for the sparse model trained on residuals.

The result is in the bulk of: 
```
Sequential Model Performance: {'RMSE': 0.11682718231230123, 'Correlation': 0.7233837879784443}
```

But optimization doesn't seem to help too much. I haven't tried to swap the second model to non RF models.

------

## Submission plans:
- For leaderboard: use model trained on training set;
- For final test: use the same model trained on training and leaderboard.

### Internal scoring method
I've done R and RMSE bootstrapping for leaderboard, but perhaps we will rank models their way.

### Model candidates
(hand picked..)
1. Best base RF model, deepnose + selected mordred 
2. Best meta RF model: RF on RFs on i) (dense) deepnose + selected mordred and ii) (sparse) leffingwell + morgan
3. Best sequential RF model: RF of sparse trained on RF on dense's residual.

Not sure what we judge on; since for meta models, polynomial ridge perform better in cross fold validation. 

------


## TO-DO:

**Priority**:

#### TO-DOs 07/24/24:
1. investigate new sparse optimization round for Leffingwell_96; what was going on 
	- looks in bulk similar to what we had; some stocasticity I guess.
2. figure out the wierd k-fold CV does not predict whole data nor leaderboard effect 
	- my current take is that dataset is extremely heterogenous, and some fold just perform very off; 
	- the leaderboard data is in the range favoring more the RF-RF stacking
	- the leaderboard data's distance to the training is not significantly different from training to training.
3. split the the meta v.s. sequential model to two notebooks [done]
4. incoporate bootstrapping in reporting leaderboard results (done, my way, but need to hear what they'd score it too)
5. the new features Cyrille provided 


1. code for making feature generation and stacking more compact for testing 
2. mix and match 
3. record performance comprehensively and carry out leaderboard testing


**Prvious left**: 

- The relationship between the value obtained from different paradigm
- Try distance (1. difference both ways; 2. difference + average)
	- tried the difference both ways.
- Mordred versions community: https://github.com/JacksonBurns/mordred-community
- MLP or SVM regression 
- Try Morgan once Cyrille shared with me Morgan + Leffingwell? Or others could try it too. But need to think about how to reduce dimensionality
- Try the Snitz normalization
- Plot std by bootstrapping
- Projection using SVD 

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
2. experiment 2 :white_check_mark:
3. experiment 3, the perfume one (if the CIDs are available)
4. experiment 6 :white_check_mark:



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