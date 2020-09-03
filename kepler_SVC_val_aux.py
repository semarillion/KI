from sklearn.svm import SVC

def SimpleGridSearch(X_train,y_train,X_test,y_test):
    best_score = 0
    cnt = 0
    for gamma in [0.001,0.01,0.1,1,10,100]:
        for C in [0.001,0.01,0.1,1,10,100]:
            cnt+=1
            print(cnt)
            svc= SVC(C=C, gamma=gamma).fit(X_train, y_train)
            svc_score=svc.score(X_test, y_test)
            if svc_score>best_score:
                best_score=svc_score
                best_parameters={'C':C,'gamma':gamma}
    return (best_score,best_parameters)
