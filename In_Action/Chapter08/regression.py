#coding:utf-8
from numpy import *
import time
import json
import requests
import md5
import re




APP_KEY = 'C2B94AB52B1BF72AEA5F5D89DD46BCC8'
APP_SECRETE = 'f2d4edeab7d44bb38fb793513ff4a52c'
BASE_API_URL = 'https://api.jd.com/routerjson'
API_VERSION = '2.0'

KEY_JSON = '360buy_param_json'
KEY_TIME = 'timestamp'
KEY_VERISION = 'v'
KEY_APP_KEY = 'app_key'
KEY_METHOD = 'method'

ID_X0 = 'x0'
ID_CPU = '94'
ID_CPU_HZ = '95'
ID_DISPLAY_SIZE = '100'
ID_DISPLAY_RES = '102'
ID_SGRAM = '108'
ID_HD_SIZE = '109'
ID_HD_SPEED = '110'
ID_RAM = '112'

# 需求特征列表[x0=1(偏置),CPU,CPU频率，显示器尺寸，显示器分辨率，显存容量，硬盘大小，硬盘速度，内存]
needFeature = [
ID_X0,
ID_CPU ,
ID_CPU_HZ ,
ID_DISPLAY_SIZE ,
ID_DISPLAY_RES ,
ID_SGRAM ,
ID_HD_SIZE ,
ID_HD_SPEED ,
ID_RAM 
]

# 初始化请求字典
reqDict = {
KEY_TIME:'',
KEY_JSON:'',
KEY_APP_KEY:APP_KEY,
KEY_METHOD:'',
KEY_VERISION:API_VERSION
}

# 标准线性回归
def standRegres(xArr,yArr):
    # 样本矩阵
    xMat = mat(xArr)
    # 标签
    yMat = mat(yArr).T
    # 正规化方程
    xTx = xMat.T*xMat
    if linalg.det(xTx) == 0.0:
        print "行列式不能为0，错误！"
        return
    # 计算回归系数
    ws = xTx.I * (xMat.T*yMat)
    return ws

# 岭回归函数
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + eye(shape(xMat)[1])*lam
    if linalg.det(denom) == 0.0:
        print "行列式不能为0，错误！"
        return
    ws = denom.I * (xMat.T*yMat)
    return ws
    
# 不同lambda的岭回归测试
def ridgeTest(xArr,yArr):
    xMat = mat(xArr); yMat=mat(yArr).T
    numTestPts = 30
    wMat = zeros((numTestPts,shape(xMat)[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat

# 获得预测误差
def getErrors(wMat,testX,testY):
    xMat = mat(testX);yMat = mat(testY).T
    predicY = wMat*(xMat.T)
    m = shape(testX)[0]
    numIters = shape(wMat)[0]
    errors = []
    for i in range(numIters):
        errors.append( sum(predicY[i,:]-yMat)**2/m )
    return errors

# 定义获取dell笔记本信息
def initSample(pageSize):
    # 初始化样本
    sampleX= [] ; sampleY= []
    for i in range(1,7):
         # 获得当前时间
        nowTime = time.strftime('%Y-%m-%d %X', time.localtime( time.time() ) )
        reqDict[KEY_TIME] = nowTime
        laptops = getLaptops(i,pageSize)
        #遍历笔记本数组,初始化样本矩阵以及标签矩阵
        for i in range(len(laptops)):
            curItem = laptops[i]
            # 获取京东价格 
            jdPrice = curItem['jdPrice']
            # 获得市场价格
            martPrice = curItem['martPrice']
            # 获得笔记本Id
            skuId = curItem['skuId']
            # 获得笔记本特征
            features = getLaptopParam(skuId)
            if(features):
                features.append(float(martPrice))
                sampleX.append(features)
                sampleY.append(jdPrice)
    return sampleX,sampleY

# 获得测试集合
def getTrainSample(sampleX,sampleY):
    trainX = [];trainY = []
    m = len(sampleX)
    numTrain = int(round(m*0.6))
    trainX = sampleX[0:numTrain]
    trainY = sampleY[0:numTrain]
    return trainX,trainY

# 定义获得测试集合
def getTestSample(sampleX,sampleY):
    testX = [];testY = []
    m = len(sampleX)
    numTest = int(round(m*0.2))
    offset = int(round(m*0.6))
    testX = sampleX[offset:offset+ numTest]
    testY = sampleY[offset:offset+ numTest]
    return testX,testY

# 获得交叉验证集合
def getCrossValidationSample(sampleX,sampleY):
    crossX = [];crossY = []
    m = len(sampleX)
    offset = int(round(m*0.6)+round(m*0.2))
    crossX = sampleX[offset:]
    crossY = sampleY[offset:]
    return crossX,crossY


# ----------------------------获得笔记本列表
# - page: 当前页码
# - pageSize: 当前页面容量
def getLaptops(page,pageSize):
    # 先获得所有笔记本
    reqDict[KEY_METHOD] = 'jingdong.ware.product.search.list.get'
    reqDict[KEY_JSON] = '{"isLoadAverageScore":"false","isLoadPromotion":"false","sort":"1","page":"%d","pageSize":"%d","keyword":"联想笔记本","client":"android"}' % (page,pageSize)
    # 获得笔记本 url
    url = generateUrl(reqDict)
    # 请求api 并且获得回调json
    req = requests.get(url)
    req.encoding = 'utf-8'
    result =  req.json()
    # 得到dell 笔记本数组
    laptops =  result['jingdong_ware_product_search_list_get_responce']['searchProductList']['wareInfo']
    return laptops

# 根据id获得笔记本参数
def getLaptopParam(skuId):
    # 先获得笔记本规格属性集
    reqDict[KEY_JSON] = '{"skuid":"'+skuId+'"}'
    reqDict[KEY_METHOD] = 'jingdong.new.ware.productsortatt.get'
    url = generateUrl(reqDict)
     # 请求api 并且获得回调json
    req = requests.get(url)
    req.encoding = 'utf-8'
    result =  req.json()
    params = result['jingdong_new_ware_productsortatt_get_responce']['resultset']
    features = []
    features.append(1.0)
    for i in range(len(params)):
        curDict = params[i]
        attId = str(curDict['attId'])
        valueId = str(curDict['valueId'])
        if(valueId == '583336'):
            continue
        if (attId in needFeature):
            if(curDict.has_key('remark') and curDict['valueId']==0):
                feature = curDict['remark']
            else:
                feature = getValueByAtt(attId,valueId,nowTime)
            if(feature == None):
                return None
            # 得到有效feature
            arr = getValidFeature(feature,attId)
            if( arr == None):
                return None
            else:
                features.insert(arr[1],arr[0])
    if(len(features) != len(needFeature)):
        return None
    else:
        return features

# 获得对应属性的值
def getValueByAtt(attId,valueId,timestamp):
    reqDict[KEY_JSON]  = '{"id":"%s"}' % (attId)
    reqDict[KEY_METHOD] = 'jingdong.new.ware.AttributeValues.query'
    url = generateUrl(reqDict)
     # 请求api 并且获得回调json
    req = requests.get(url)
    req.encoding = 'utf-8'
    result =  req.json()
    atts = result['jingdong_new_ware_AttributeValues_query_responce']['resultset']
    for i in range(len(atts)):
        curDict = atts[i]
        if(str(curDict['valueId'])==valueId):
            return curDict['valueName']

# ----------------------------产生签名函数
# -params:请求参数
def generateSign(params):
    params = sorted(params.iteritems(),key=lambda d:d[0])
    src = '' + APP_SECRETE
    for i in range(len(params)):
        src = src+params[i][0]+params[i][1]
    src = src + APP_SECRETE
    md5Ins = md5.new()
    md5Ins.update(src);
    sign = md5Ins.hexdigest();
    sign = sign.upper()
    return sign


# 产生apiurl
def generateUrl(reqDict):
    sign = generateSign(reqDict)
    url = BASE_API_URL+'?'+KEY_VERISION+'=2.0&'+KEY_METHOD+'='+reqDict[KEY_METHOD]+'&'+KEY_APP_KEY+'='+APP_KEY+'&'+KEY_JSON+'='+reqDict[KEY_JSON]+'&'+KEY_TIME+'='+reqDict[KEY_TIME]+'&sign=' +sign
    return url

#  通过正则表达式提炼特征
def getValidFeature(feature,attId):
    feature = feature.encode('utf-8')
    if(attId == ID_CPU ):
        m = re.search(r'i\d',feature,re.I)
        if(m):
            return float(m.group().upper().replace('I','')),needFeature.index(ID_CPU)
        else:
            return None

    if(attId == ID_CPU_HZ ):
        m = re.search(r'\d+\.\d+',feature,re.I)
        if(m):
            return float(m.group()),needFeature.index(ID_CPU_HZ)
        else:
            return None

    if(attId == ID_DISPLAY_SIZE ):
        m = re.search(r'\d+\.?\d?',feature)
        if(m):
            return float(m.group()),needFeature.index(ID_DISPLAY_SIZE)
        else:
            return None

    if(attId == ID_DISPLAY_RES ):
        feature = feature.replace(' ','')
        arr =  re.split('x|×',feature)
        if(len(arr) == 2):
            ratio = float(arr[0])*float(arr[1])
            return ratio,4
        else:
            return None

    if(attId == ID_SGRAM ):
        m = re.search(r'\d+',feature)
        if(m):
            ram = m.group()
            if(len(ram) == 1):
                ram = float(ram)*1024
            else:
                ram = float(ram)
            return ram,needFeature.index(ID_SGRAM)
        else:
            return None
            
    if(attId == ID_HD_SIZE ):
        m = re.search(r'\d+',feature)
        if(m):
            size = m.group()
            if(len(size)==1):
                size = float(size)*1000*1024
            else:
                size = float(size)
            return size,needFeature.index(ID_HD_SIZE)
        else:
            return None
            
    if(attId == ID_HD_SPEED ):
        m = re.search(r'\d+',feature)
        if(m):
            return float(m.group()),needFeature.index(ID_HD_SPEED)
        else:
            return None
            
    if(attId == ID_RAM ):
        m = re.search(r'\d+',feature)
        if(m):
            return float(m.group()),needFeature.index(ID_RAM)
        else:
            return None
            
