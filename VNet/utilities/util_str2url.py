import urllib.parse as urllibparse

pageoption = [{"plugin":"images",
                   "tag":"max_GT_test",
                   "runId":"defaultExperimentId/.",
                   "sample":0},
                  {"plugin":"images",
                   "tag":"max_GT_train",
                   "runId":"defaultExperimentId/.",
                   "sample":0},
                  {"plugin":"images",
                   "tag":"max_GT_val",
                   "runId":"defaultExperimentId/.",
                   "sample":0},
                  {"plugin":"images",
                   "tag":"max_prediction_test",
                   "runId":"defaultExperimentId/.",
                   "sample":0},
                  {"plugin":"images",
                   "tag":"max_prediction_train",
                   "runId":"defaultExperimentId/.",
                   "sample":0},
                  {"plugin":"images",
                   "tag":"max_prediction_val",
                   "runId":"defaultExperimentId/.",
                   "sample":0}
                 ]

def str2url(s):
    sout = ""
    for c in s:
        if c=="'":
            c = '"'
        if c!='"' and c!='_' and c!='%':
            sout += urllibparse.quote(c)
        else:
            sout += c
    return sout

def url2str(s):
    return urllibparse.unquote(s)

def getTBurl(ipaddr="localhost"):
    pageoptionstr = str(pageoption).replace(" ","").replace("'","\"").replace("/","%2F")
    return "http://"+ipaddr+":6006/?pinnedCards=" + str2url(pageoptionstr) + "#timeseries"

if __name__ == '__main__':
    
    pageoptionstr = str(pageoption).replace(" ","").replace("'","\"").replace("/","%2F")
    print("http://localhost:6006/?pinnedCards=" + pageoptionstr+ "#timeseries")
    print("\n")
    print("http://localhost:6006/?pinnedCards=" + str2url(pageoptionstr) + "#timeseries")
    print("\n")
    print('http://localhost:6006/?pinnedCards=%5B%7B"plugin"%3A"images"%2C"tag"%3A"max_GT_test"%2C"runId"%3A"defaultExperimentId%2F."%2C"sample"%3A0%7D%2C%7B"plugin"%3A"images"%2C"tag"%3A"max_GT_train"%2C"runId"%3A"defaultExperimentId%2F."%2C"sample"%3A0%7D%2C%7B"plugin"%3A"images"%2C"tag"%3A"max_GT_val"%2C"runId"%3A"defaultExperimentId%2F."%2C"sample"%3A0%7D%2C%7B"plugin"%3A"images"%2C"tag"%3A"max_prediction_test"%2C"runId"%3A"defaultExperimentId%2F."%2C"sample"%3A0%7D%2C%7B"plugin"%3A"images"%2C"tag"%3A"max_prediction_train"%2C"runId"%3A"defaultExperimentId%2F."%2C"sample"%3A0%7D%2C%7B"plugin"%3A"images"%2C"tag"%3A"max_prediction_val"%2C"runId"%3A"defaultExperimentId%2F."%2C"sample"%3A0%7D%5D#timeseries')
    print("\n")
    print(str2url('http://localhost:6006/?pinnedCards=[{"plugin":"images","tag":"max_GT_test","runId":"defaultExperimentId/.","sample":0},{"plugin":"images","tag":"max_GT_train","runId":"defaultExperimentId/.","sample":0},{"plugin":"images","tag":"max_GT_val","runId":"defaultExperimentId/.","sample":0},{"plugin":"images","tag":"max_prediction_test","runId":"defaultExperimentId/.","sample":0},{"plugin":"images","tag":"max_prediction_train","runId":"defaultExperimentId/.","sample":0},{"plugin":"images","tag":"max_prediction_val","runId":"defaultExperimentId/.","sample":0}]#timeseries'))