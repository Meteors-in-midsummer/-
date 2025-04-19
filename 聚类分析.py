import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("E:/大三上/点宽杯/111.csv")
data = data.dropna()
# 用户基本属性分析
data['年龄段'] = pd.cut(data['年龄'], bins=[0, 18, 24, 34, 60, 100], labels=['<18', '18-24', '25-34', '35-59', '60+'])
data['性别'] = data['性别'].astype('category')
user_behavior_counts = data.groupby('用户ID')['行为'].value_counts()
# 用户消费能力分析
scaler = StandardScaler()
data['经济权重标准化'] = scaler.fit_transform(data[['经济权重']])
kmeans = KMeans(n_clusters=4, n_init=10, random_state=0)
kmeans.fit(data[['经济权重标准化']])
region_mapping = {
    '北京': '华北', '天津': '华北', '河北': '华北', '山西': '华北', '内蒙古': '华北',
    '上海': '华东', '江苏': '华东', '浙江': '华东', '安徽': '华东', '福建': '华东', '江西': '华东', '山东': '华东',
    '广东': '华南', '广西': '华南', '海南': '华南', '香港': '华南', '澳门': '华南',
    '河南': '华中', '湖北': '华中', '湖南': '华中',
    '重庆': '西南', '四川': '西南', '贵州': '西南', '云南': '西南', '西藏': '西南',
    '陕西': '西北', '甘肃': '西北', '青海': '西北', '宁夏': '西北', '新疆': '西北',
    '台湾': '华东','黑龙江': '东北', '吉林': '东北', '辽宁': '东北'}
data['区域'] = data['省份'].map(region_mapping)
data['消费能力等级'] = kmeans.labels_
# 构建用户画像
user_profile = data.groupby('用户ID').agg({
    '年龄段': 'first',
    '性别': 'first',
    '区域': 'first',
    '商品类型': lambda x: ' '.join(x.unique()),  
    '具体商品': lambda x: ' '.join(x.unique())   
}).reset_index()

user_profile.head()
user_profile.info()
