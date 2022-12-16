# caurlaeogus031128_open
openSW final 20226275 김대현
## What i did in my project?
가장 먼저 데이터를 살펴보았다. 데이터는 64*64의 이미지 데이터였고, no_tumor 라벨의 데이터가 가장 적었다. 또한 전체적인 데이터의 수도 적었다.
따라서 전처리를 진행하여 데이터의 수를 늘려 학습을 진행하고자 하였다.
## Explain the training data set
이 데이터 셋은 glioma_tumor, meningioma_tumor, no_tumor, pituitary_tumor 라는 4개의 라벨을 가지고 있다. 각각은 이미지 데이터로 우리는 이를 64*64 이미지로 받아왔다. 
## Explain the algorithm you choose
KNeighborsClassifier, SVC, ExtraTreesClassifier, VotingClassifier 의 4가지 모델을 사용하였다.
먼저 KNeighborsClassifer는 k-nearest neighbors vote에 기반하여 분류를 하는 skleanr의 분류기이다.

둘 째로 SVC는 Support Vector Machines의 모델 중 하나이다.

셋 째로 ExtraTreesClassifier는 랜덤 포레스트 모델을 개량한 것으로, 각 포레스트 트리의 후보 특성을 랜덤으로 분할하여 무작위성을 증가시키는 모델이다.

마지막으로 VotingClassifier는 내부에 있는 모델의 판단에 따라 최종 판단을 내리는 모델이다.
## Explain hyper-parameter of the function
KNN 모델의 하이퍼파라미터는 아래와 같으며,
```python
KNeighborsClassifier(n_neighbors=5, *, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
```
나는 그 중 n_neighbors를 1로 수정하였다.

둘 째로 SVC 모델의 하이퍼파라미터는 아래와 같으며,
```python
class sklearn.svm.SVC(*, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', break_ties=False, random_state=None)
```
나는
 ```python
sklearn.svm.SVC(gamma = 1.2, probability=True, random_state = 521 )와 같이 수정하였다.
```
셋 째로 SVC 모델의 하이퍼파라미터는 아래와 같으며,
 ```python
class sklearn.ensemble.ExtraTreesClassifier(n_estimators=100, *, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=False, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
```
나는
```python
sklearn.ensemble.ExtraTreesClassifier(n_estimators = 400, max_depth = 24, max_leaf_nodes = 1000, random_state = 600, n_jobs = -1)
```
와 같이 수정하였다.

마지막으로 VotingClassifier 모델의 하이퍼파라미터는 아래와 같으며, 
```python
VotingClassifier(estimators, *, voting='hard', weights=None, n_jobs=None, flatten_transform=True, verbose=False)
```
나는 voting = 'hard' 를 선택하였다. 

## how to optimize hyper-parameter?
나는 주로 gridsearchCV 와 RandomizedsearchCV를 이용하였고 그 둘을 이용하여 정확도를 높일 수 있는 하이퍼 파라미터를 빠르게 찾을 수 있었다.
