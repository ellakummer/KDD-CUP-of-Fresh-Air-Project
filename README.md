# KDD-CUP-of-Fresh-Air-Project
13X011 - UNIGE

Dataset missing because too heavy : 
London_historical_meo_grid.csv
Beijing_historical_meo_grid.csv


Possible models : (we need supervised learning)
https://machinelearningmastery.com/a-tour-of-machine-learning-algorithms/

pas convaincue :

https://fr.wikipedia.org/wiki/Machine_%C3%A0_vecteurs_de_support
-> pour classes prédéfinies... :( => càd ? ah nevermind

https://en.wikipedia.org/wiki/Recurrent_neural_network

https://towardsdatascience.com/multinomial-naive-bayes-classifier-for-text-analysis-python-8dd6825ece67
-> A Naive Bayes model assumes that each of the features it uses are conditionally independent of one another given some class. (so not for us=

https://fr.wikipedia.org/wiki/Lasso_(statistiques)
-> soucis : lorsque notamment la dimension est trop élevée (dead pour nous quoi) 

https://fr.wikipedia.org/wiki/R%C3%A9gression_locale
-> soucis :  relativement intensive en calculs, ce qui peut poser problème lorsque le jeu de données utilisé est très important. (dead pour nous quoi) 


à tester à mon avis : 

https://machinelearningmastery.com/bagging-and-random-forest-ensemble-algorithms-for-machine-learning/
-> pour vue générale ? ... partout ils en parlent pour la "météo"
-> pas mal mais implique la multiplication de données pour meilleures estimation => on a 10M de données (par ville) ... nos pc vont mourir

https://en.wikipedia.org/wiki/Gradient_boosting ouais ouais j'ai tout compris tkt -> (je m'en occupe !)
https://www.youtube.com/watch?v=3CC4N4z3GJc
https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting

https://fr.wikipedia.org/wiki/R%C3%A9gression_multivari%C3%A9e_par_spline_adaptative -> c'est pas pour des features indépendantes ? sinon stylé

https://fr.wikipedia.org/wiki/R%C3%A9gularisation_de_Tikhonov 

https://en.wikipedia.org/wiki/Elastic_net_regularization ouais ouais j'ai tout compris tkt bis
https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html -> vive sckit

https://en.wikipedia.org/wiki/Least-angle_regression -> not bad i think ( if not ( ouais ouais j'ai tout compris tkt) )

https://fr.wikipedia.org/wiki/R%C3%A9seau_bay%C3%A9sien 
-> "L'intérêt particulier des réseaux bayésiens est de tenir compte simultanément de connaissances a priori d'experts (dans le graphe) et de l'expérience contenue dans les données." => super bien pour le meteo grid et observable non ? problème -> tables de probabilités (base + estimée) de bâtard ...

https://fr.wikipedia.org/wiki/Mod%C3%A8le_additif_g%C3%A9n%C3%A9ralis%C3%A9 this one !! 

Python librairy : 
scikit-learn

inspiration ? 
https://stackoverflow.com/questions/41925157/logisticregression-unknown-label-type-continuous-using-sklearn-in-python
