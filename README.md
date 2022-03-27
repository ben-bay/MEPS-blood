# MEPS disease classifier

This work predicts whether patients have high blood pressure based on their medication history and a few personal features.

I chose to predict `highBPDiagnosed` since it is the least imbalanced of all the labels. I made a sparse multi-hot encoding vector for each patient, showing which of the most common medications they used. I included a few other patient features, then trained binary classification models.

The best solution I found is a stacked ensemble model that achieves F1 of 0.813. The DNN appeared to slightly detract, so running it is optional.

	Learner                  Accuracy    AUC         F1
	----------------------   ----------  --------    --------
	Random baseline          0.50389     0.503754    0.481748
	Logistic regression      0.831016    0.824917    0.803033
	Gradient-boosted trees   0.834587    0.829139    0.808899
	Random forest            0.813034    0.810319    0.79241
	Stacked ensemble         0.837011    0.832112    0.813103

Takes between 8 seconds and 2.5 minutes on M1 Macbook Pro, depending on command line arguments. Run like this:

	python3 run\_solution.py [--base] [--meds] [--stack] [--deep] [--help]
