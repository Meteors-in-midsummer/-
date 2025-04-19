# 电商平台用户行为分析与可视化
基于das平台电商数据，深入挖掘用户行为和市场趋势，提升用户体验和销售额。
# 电商平台用户行为分析与销售优化
# 项目周期：2024年3月 - 2024年6月  
# 技术栈：Python (Pandas/Matplotlib/Scikit-learn)、Excel
# 关键词：用户画像、RFM模型、漏斗分析、协同过滤推荐 
## 项目背景
# 针对某电商平台（日均UV 50万+），分析用户从浏览到支付的完整行为路径，定位转化率瓶颈并提出数据驱动的运营优化方案，项目最终获得一等奖
导入pandas库，命名为pd
导入numpy库并命名为np
从 datetime 导入 timedelta
customers = pd.read_csv(r'E:\大三上\点宽杯\用户信息(id所在地年龄性别权益）.csv')
用户行为 = pd.read_csv(r'E:\大三上\点宽杯\用户行为商品时间备注.csv')
products = pd.read_csv(r'E:\大三上\点宽杯\商品（id类型具体商品成本单价库存）.csv')
promotions = pd.read_csv(r'E:\大三上\点宽杯\日期周末假期折扣量.csv')
locations = pd.read_csv(r'E:\大三上\点宽杯\省份城市经济权重.csv')
# 对地区信息表进行数据清洗
print("数据概况：")
打印(locations.info())
print("\n数据的前几行：")
打印(地点.头部())
print("\n缺失值统计：")
print(locations.isnull().sum())
locations['经济权重'] = locations['经济权重'].fillna(locations['经济权重'].median())
locations.dropna(subset=['省份', '城市'], inplace=True)
print("\n重复值统计：")
打印(locations.duplicated().sum())
locations = locations.drop_duplicates()
locations['省份'] = locations['省份'].str.strip().str.title()
locations['城市'] = locations['城市'].str.strip().str.title()
print("\n经济权重描述：")
打印(地点['经济权重'].描述())
# 去除负值
locations = locations[locations['经济权重'] >= 0]
locations = locations[locations['经济权重'] <= locations['经济权重'].quantile(0.95)]
# 清洗之后的文件保存为111
cleaned_file_path = r"E:\大三上\点宽杯\省份城市经济权重111.csv"
locations.to_csv(清理后的文件路径, index=False)
print(f"\n数据清洗完成，已保存为：{cleaned_file_path}")
# 对用户信息表进行数据清洗
用户行为['时间'] = pd.to_datetime(用户行为['时间'])
customers.fillna({'年龄': 0, '性别': '未知', '权益': '非VIP'}, inplace=True)
最小年龄 = 0
最大年龄 = 120
离群值 = (顾客['年龄'] < 最小年龄) | (顾客['年龄'] > 最大年龄)

# 用中位数年龄替换异常值
median_age = customers['年龄'].median()
customers.loc[离群值, '年龄'] = 中位年龄
customers.to_csv(r'E:\大三上\点宽杯\用户信息异常值清洗.csv', index=False)
products.fillna({'成本': 0, '单价': 0, '库存': 0}, inplace=True)
promotions.fillna({'折扣量': 0}, inplace=True)
用户行为产品 = pd.merge(用户行为, 产品, on='商品ID', how='left')

# 解释了用户购买了什么商品类型下的什么东西
#user_behavior_products.to_csv(r"E:\大三上\点宽杯\用户行为买了什么商品.csv", index=False)

# 销售规律分析
# 按商品类型和日期聚合销售数据
# 相关性分析
sales_data = user_behavior_products[user_behavior_products['行为'] == '购买'].groupby(['商品类型', pd.Grouper(key='时间', freq='ME')])['单价'].sum().reset_index()

sales_data.rename(columns={'单价': '月销售额'}, inplace=True)#给出每种商品类型的月销售总额

sales_data_corr = sales_data.groupby('商品类型')['月销售额'].corr(promotions.groupby('日期')['折扣量'].mean().reset_index(drop=True))
print("各类商品月销售额与平均折扣量的相关性：")
print(sales_data_corr)
# 计算用户行为指标
#1.活跃度（DAU，Daily Active Users）：每日

daily_active_users = user_behavior['时间'].value_counts().sort_index()
daily_active_users_df = pd.DataFrame(list(daily_active_users.items()), columns=['时间', '活跃度'])
daily_active_users = user_behavior.groupby(user_behavior['时间'].dt.date)['用户ID'].nunique()


#3. 跳失率（Bounce Rate）：仅仅浏览一个页面就离开的用户比例

bounce_users_daily = user_behavior[user_behavior['行为'] == '浏览'].groupby('时间')['用户ID'].apply(lambda x: x.unique())
total_users_daily = user_behavior.groupby('时间')['用户ID'].nunique()# 计算每天的总用户数
bounce_rate_daily = {}# 计算每日跳失率
for date, users in bounce_users_daily.items():
    total_users = total_users_daily.loc[date]
    bounce_rate = len(users) / total_users if total_users > 0 else 0
    bounce_rate_daily[date] = bounce_rate
bounce_rate_daily_df = pd.DataFrame(list(bounce_rate_daily.items()), columns=['时间', '跳失率'])

#4. 用户转化率（Conversion Rate）：执行购买行为的用户占所有用户的比例

purchasing_users = user_behavior[user_behavior['行为'] == '购买']['用户ID'].unique()
conversion_rate = len(purchasing_users) / len(user_behavior['用户ID'].unique()) if len(user_behavior['用户ID'].unique()) > 0 else 0

#5. 用户增长率（User Growth Rate）：新增用户数量与前一天新增用户数量的比例

user_growth_rate = []
all_new_users = user_behavior.groupby(user_behavior['时间'].dt.date)['用户ID'].nunique()
for i in range(1, len(all_new_users)):
    new_users_day0 = all_new_users.iloc[i - 1]
    new_users_day1 = all_new_users.iloc[i]
    growth_rate = (new_users_day1 - new_users_day0) / new_users_day0 if new_users_day0 > 0 else 0
user_growth_rate.append(growth_rate)

#6. 客户流失率（Customer Churn Rate）：30天内未再活跃的用户比例

last_active_date = user_behavior.groupby('用户ID')['时间'].max().reset_index()
inactive_users = last_active_date[last_active_date['时间'].dt.date < (max(user_behavior['时间'].dt.date) - timedelta(days=30))]['用户ID'].unique()
churn_rate = len(inactive_users) / len(user_behavior['用户ID'].unique()) if len(user_behavior['用户ID'].unique()) > 0 else 0

#展示用户行为指标

print("\n用户行为指标：")
print("活跃度（DAU）：", daily_active_users)  # 仅展示最后一天的DAU作为示例
print("用户转化率：", conversion_rate)
print("用户增长率：", np.mean(user_growth_rate))
print("客户流失率：", churn_rate)

#展示每个用户最常购买的商品类别

user_behavior_purchase = user_behavior[user_behavior['行为'] == '购买']
user_behavior_purchase = pd.merge(user_behavior_purchase, products[['商品ID', '商品类型']], on='商品ID', how='left')
user_behavior_purchase_category = user_behavior_purchase.groupby(['用户ID', '商品类型'])['商品ID'].count().reset_index()
user_behavior_purchase_category.rename(columns={'商品ID': '购买次数'}, inplace=True)
user_most_bought_category = user_behavior_purchase_category.loc[user_behavior_purchase_category.groupby('用户ID')['购买次数'].idxmax()]
user_most_bought_category = user_most_bought_category[['用户ID', '商品类型']]
print("\n每个用户最常购买的商品类别：")
print(user_most_bought_category.head())

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
import seaborn as sns
from scipy import stats

user_behavior = pd.read_csv(r"E:\大三上\点宽杯\用户行为商品时间备注.csv")
products = pd.read_csv(r"E:\大三上\点宽杯\商品（id类型具体商品成本单价库存）.csv")
promotions = pd.read_csv(r"E:\大三上\点宽杯\日期周末假期折扣量.csv")
promotions.rename(columns={'日期': '时间'}, inplace=True)
promotions['时间'] = pd.to_datetime(promotions['时间'])
# R的时间间隔计算
user_behavior_products = pd.merge(user_behavior, products, on='商品ID', how='left')
user_behavior_products['时间'] = pd.to_datetime(user_behavior_products['时间'])
user_behavior_products_purchases = user_behavior_products[user_behavior_products['行为'] == '购买']
user_behavior_products_recent_purchases = user_behavior_products_purchases.groupby('用户ID')['时间'].max().reset_index()
target_date = pd.to_datetime('2024-11-15')
user_behavior_products_recent_purchases['R'] = (target_date - user_behavior_products_recent_purchases['时间']).dt.days
R = user_behavior_products_recent_purchases[['用户ID', 'R']]
# F的下单次数计算
order_counts = user_behavior_products_purchases.groupby('用户ID').size()
F = order_counts.reset_index(name='F')
# M的平均消费金额计算
df_final = pd.merge(user_behavior_products_purchases, promotions, on='时间', how='left')
df_final['折扣后价格'] = df_final['单价'] * df_final['折扣量']
average_spending = df_final.groupby('用户ID')['折扣后价格'].mean()
M = average_spending.reset_index(name='M')
# 合并RFM序列
RFM = R[["用户ID", "R"]].merge(F, on="用户ID").merge(M, on="用户ID")
pd.set_option('display.unicode.east_asian_width', True)
# 重设index
RFM.set_index("用户ID", inplace=True)


# 将RFM模型的原始值处理成对应的等级
def RFM_R(value):
    if value <= 75:
        return 5
    elif value <= 110:
        return 4
    elif value <= 200:
        return 3
    elif value <= 350:
        return 2
    return 1


def RFM_F(value):
    if value <= 3:
        return 1
    elif value <= 10:
        return 2
    elif value <= 20:
        return 3
    elif value <= 30:
        return 4
    return 5


def RFM_M(value):
    if value < 550:
        return 1
    elif value < 600:
        return 2
    elif value < 650:
        return 3
    elif value < 720:
        return 4
    return 5


RFM['R'] = RFM.R.apply(RFM_R)
RFM['F'] = RFM.F.apply(RFM_F)
RFM['M'] = RFM.M.apply(RFM_M)
column_means = RFM.mean()
RFM_model = RFM.apply(lambda x: x.apply(lambda xi: '高' if xi > column_means[x.name] else '低'))
RFM_model['Tag'] = RFM_model.sum(axis=1)
RFM_model['Tag'] = RFM_model.Tag.astype('category').cat.reorder_categories(
    ['高高高', '高低高', '低高高', '低低高', '高高低', '高低低', '低高低', '低低低'])
RFM_model['Tag'] = RFM_model['Tag'].cat.rename_categories(
    {'高高高': '高价值客户', '高低高': '重点发展客户', '低高高': '重点保持客户', '低低高': '重点挽留客户',
     '高高低': '一般价值客户', '高低低': '一般发展客户', '低高低': '一般保持客户', '低低低': '潜在客户'})
result = RFM_model.Tag.value_counts().sort_index()
font = {"family": "Microsoft Yahei"}
plt.rc("font", **font)
result.plot(kind='bar')
plt.title('用户分组——RFM模型')
dm = pd.merge(df_final, RFM_model, on='用户ID', how='left')
consumption_amount = dm.groupby('Tag', observed=True)['折扣后价格'].sum()
purchase_quantity = dm.groupby('Tag', observed=True)['商品ID'].size()
customer_count = dm['Tag'].value_counts()
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('RFM每层客户各项指标对比图', fontsize=16)
axs[0, 0].pie(consumption_amount, labels=consumption_amount.index, autopct='%1.1f%%', startangle=140)
axs[0, 0].set_title('消费金额对比')
axs[0, 1].pie(purchase_quantity, labels=purchase_quantity.index, autopct='%1.1f%%', startangle=140)
axs[0, 1].set_title('购买商品总量对比')
axs[1, 0].pie(customer_count, labels=customer_count.index, autopct='%1.1f%%', startangle=140)
axs[1, 0].set_title('客户数量对比')
axs[1, 1].axis('off')
plt.show()


# K-means聚类分析
def draw_plots(data, title, row, col, index):
    fig, axs = plt.subplots(row, col, figsize=(6, 6), dpi=200)
    plt.subplots_adjust(wspace=0.6, hspace=0.6)
    plt.style.use('fivethirtyeight')
    for i, j in enumerate(['R', 'F', 'M']):
        sns.histplot(data[j], kde=True, ax=axs[0, i], color='blue')
        (mu, sigma) = stats.norm.fit(data[j])
        axs[0, i].plot(data[j], stats.norm.pdf(data[j], mu, sigma), 'r', linewidth=2)
        axs[0, i].legend([r'Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f})'.format(mu, sigma)], loc='best',
                         fontsize=3)
        axs[0, i].set_ylabel('数量', fontsize=3)
        axs[0, i].set_title('{} 频数图'.format(j), fontsize=6)
        stats.probplot(data[j], plot=axs[1, i])
        axs[1, i].set_title('{} 概率图'.format(j), fontsize=6)


draw_plots(RFM, 'RFM Distributions', 2, 3, 0)
pd.DataFrame([i for i in zip(RFM.columns, RFM.skew(), RFM.kurt())], columns=['特征', '偏度', '峰度'])
from sklearn.preprocessing import StandardScaler

standard_scaler = StandardScaler()
standard_scaler.fit(RFM)
data_scaler = standard_scaler.transform(RFM)
k_data_scaler = pd.DataFrame(data_scaler, columns=['R', 'F', 'M'], index=RFM.index)
from sklearn.cluster import KMeans

inertia_list = []
for k in range(1, 10):
    model = KMeans(n_clusters=k, max_iter=500, n_init=10, random_state=12)
    kmeans = model.fit(k_data_scaler)
    inertia_list.append(kmeans.inertia_)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(1, 10), inertia_list, '*-', linewidth=1)
ax.set_xlabel('k')
ax.set_ylabel("inertia_score")
ax.set_title('inertia变化图')
plt.show()
# 观察上图可知K=2时有明显的拐点，但实际业务中2类并不能够很好地满足需求，所以接下来使用廓系数评估。
from sklearn import metrics

label_list = []
silhouette_score_list = []
for k in range(2, 10):
    model = KMeans(n_clusters=k, max_iter=500, n_init=10, random_state=12)
    kmeans = model.fit(k_data_scaler)
    silhouette_score = metrics.silhouette_score(k_data_scaler, kmeans.labels_)  # 轮廓系数
    silhouette_score_list.append(silhouette_score)
    label_list.append({k: kmeans.labels_})
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(2, 10), silhouette_score_list, '*-', linewidth=1)
ax.set_xlabel('k')
ax.set_ylabel("silhouette_score")
ax.set_title('轮廓系数变化图')
plt.show()
calinski_harabaz_score_list = []
for i in range(2, 10):
    model = KMeans(n_clusters=i, n_init=10, random_state=1234)
    kmeans = model.fit(k_data_scaler)
    calinski_harabaz_score = metrics.calinski_harabasz_score(k_data_scaler, kmeans.labels_)
    calinski_harabaz_score_list.append(calinski_harabaz_score)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(range(2, 10), calinski_harabaz_score_list, '*-', linewidth=1)
ax.set_xlabel('k')
ax.set_ylabel("calinski_harabaz_score")
ax.set_title('calinski_harabaz_score变化图')
# plt.show()
model = KMeans(n_clusters=4, n_init=10, random_state=12345)
kmeans = model.fit(k_data_scaler)
RFM['label'] = kmeans.labels_
k_data_scaler['label'] = kmeans.labels_
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8), dpi=200)
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=30, azim=-30)
fig.add_axes(ax)
ax1 = ax.scatter(k_data_scaler.query("label == 0").R, k_data_scaler.query("label == 0").F,
                 k_data_scaler.query("label == 0").M, edgecolor='k', color='r')
ax2 = ax.scatter(k_data_scaler.query("label == 1").R, k_data_scaler.query("label == 1").F,
                 k_data_scaler.query("label == 1").M, edgecolor='k', color='b')
ax3 = ax.scatter(k_data_scaler.query("label == 2").R, k_data_scaler.query("label == 2").F,
                 k_data_scaler.query("label == 2").M, edgecolor='k', color='c')
ax4 = ax.scatter(k_data_scaler.query("label == 3").R, k_data_scaler.query("label == 3").F,
                 k_data_scaler.query("label == 3").M, edgecolor='k', color='g')
ax.legend([ax1, ax2, ax3, ax4], ['Cluster 1', 'Cluster 2', 'Cluster 3', 'Cluster 4'])
ax.invert_xaxis()
ax.set_xlabel('R')
ax.set_ylabel('F')
ax.set_zlabel('M')
ax.set_title('K-Means Clusters')
plt.show()
# 将RFM数据和其他特征数据合并
merged_rf = pd.merge(R, F, on="用户ID")
kmeans_data = pd.merge(merged_rf, M, on="用户ID")
kmeans_data.columns = ['用户ID', '最近消费天数', '消费次数', '平均消费金额']
# 转换列为数值类型，无法转换的设置为NaN
kmeans_data['最近消费天数'] = pd.to_numeric(kmeans_data['最近消费天数'], errors='coerce')
kmeans_data['消费次数'] = pd.to_numeric(kmeans_data['消费次数'], errors='coerce')
kmeans_data['平均消费金额'] = pd.to_numeric(kmeans_data['平均消费金额'], errors='coerce')
kmeans_data.dropna(inplace=True)
kmeans = KMeans(n_clusters=4, n_init=10, random_state=0).fit(kmeans_data[['最近消费天数', '消费次数', '平均消费金额']])
kmeans_data['Cluster'] = kmeans.labels_
for cluster in range(4):
    cluster_data = kmeans_data[kmeans_data['Cluster'] == cluster]
    print(f"Cluster {cluster} statistics:")
    print(cluster_data.describe())
kmeans_data['Cluster'] = kmeans_data['Cluster'].replace(
    {2: '重要价值客户', 1: '中等价值客户', 3: '低等价值客户', 0: '潜在客户'})
consumption_amount2 = kmeans_data.groupby('Cluster')['平均消费金额'].sum()
purchase_quantity2 = kmeans_data.groupby('Cluster')['用户ID'].size()
customer_count2 = kmeans_data['用户ID'].value_counts()
fig, axs = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('K-means每层客户各项指标对比图', fontsize=16)
axs[0, 0].pie(consumption_amount2, labels=consumption_amount2.index, autopct='%1.1f%%', startangle=140)
axs[0, 0].set_title('消费金额对比')
axs[0, 1].pie(purchase_quantity2, labels=purchase_quantity2.index, autopct='%1.1f%%', startangle=140)
axs[0, 1].set_title('购买商品总量对比')
axs[1, 0].pie(customer_count2, labels=customer_count2.index, autopct='%1.1f%%', startangle=140)
axs[1, 0].set_title('客户数量对比')
axs[1, 1].axis('off')
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
import squarify
users = pd.read_csv("E:\\大三上\\点宽杯\\用户信息异常值清洗.csv")
locations = pd.read_csv("E:\\大三上\\点宽杯\\省份城市经济权重.csv")
locations.rename(columns={'城市': '用户所在地'}, inplace=True)
df_merged = pd.merge(users,locations, on='用户所在地', how='left')
user_behavior_products=pd.read_csv(r"E:\大三上\点宽杯\用户行为买了什么商品.csv")
df=pd.merge(df_merged,user_behavior_products,on='用户ID',how='left')
purchase_df = df[df['行为'] == '购买']
province_sales_count = purchase_df.groupby(['省份', '商品类型']).size().unstack(fill_value=0)
mask = province_sales_count.isnull()
sns.set(style="whitegrid")
# 商品类型销售量分布热力图
plt.figure(figsize=(12, 8))
sns.set(font="simhei")
ax = sns.heatmap(province_sales_count, mask=mask, center=0, cmap='RdBu_r', annot=True, fmt="d", cbar_kws={"shrink": .8})
ax.set_title('商品类型销售量分布热力图', fontsize=16)
ax.set_xlabel('商品类型', fontsize=12)
ax.set_ylabel('省份', fontsize=12)
ax.tick_params(axis='both', which='major', labelsize=10)
plt.show()
#商品利润点图
products = pd.read_csv(r"E:\大三上\点宽杯\商品（id类型具体商品成本单价库存）.csv")
products['利润']=products['单价']-products['成本']
plt.scatter(range(len(products)), products['利润'], alpha=0.5)
plt.xlabel('商品编号')
plt.ylabel('利润')
plt.title('商品利润点图')
plt.grid(True)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import numpy as np
plt.rcParams['font.sans-serif'] = ['SimHei']
pd.set_option('display.unicode.east_asian_width',True)
users = pd.read_csv("E:\\大三上\\点宽杯\\用户信息异常值清洗.csv")
locations = pd.read_csv("E:\\大三上\\点宽杯\\省份城市经济权重.csv")
locations.rename(columns={'城市': '用户所在地'}, inplace=True)
df_merged = pd.merge(users,locations, on='用户所在地', how='left')
user_behavior_products=pd.read_csv(r"E:\大三上\点宽杯\用户行为买了什么商品.csv")
df=pd.merge(df_merged,user_behavior_products,on='用户ID',how='left')
sales_area=df.groupby('省份')['单价'].sum()
# 各地区销售额
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(sales_area, labels=sales_area.index, autopct='%1.1f%%', startangle=90, pctdistance=0.85)
for text in texts:
    text.set_rotation(30)
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_size(6)
plt.title('各地区销售额')
# 不同商品类型销售额占比
sales_type=df.groupby('商品类型')['单价'].sum()
plt.figure(figsize=(8, 8))
plt.pie(sales_type, labels=sales_type.index, autopct='%1.1f%%', startangle=140)
plt.title('不同商品类型销售额占比')
plt.axis('equal')
#总订单占比
member_count = df[df['权益'] == '会员'].shape[0]
non_member_count = df[df['权益'] != '会员'].shape[0]
total_count = member_count + non_member_count
member_ratio = member_count / total_count
non_member_ratio = non_member_count / total_count
labels = ['会员', '非会员']
sizes = [member_ratio, non_member_ratio]
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.set_title('总订单占比')
#总消费金额占比
grouped_df = df.groupby('权益')['单价'].sum()
fig, ax = plt.subplots()
ax.pie(grouped_df, labels=grouped_df.index, autopct='%1.1f%%', startangle=90)
ax.set_title('总消费金额占比')
#会员的年龄分布
members = users[users['权益'] == '会员']
def age_group(age):
    if age < 18:
        return '未成年人'
    elif 18 <= age <= 44:
        return '年轻人'
    elif 45 <= age <= 60:
        return '中年人'
    else:
        return '老年人'
members_copy = members.copy()
members_copy['年龄组'] = members_copy['年龄'].apply(age_group)
age_group_counts = members_copy['年龄组'].value_counts()
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(age_group_counts, labels=age_group_counts.index, autopct='%1.1f%%', startangle=90, pctdistance=0.90, wedgeprops=dict(width=0.3))
for text in texts:
    text.set_rotation(35)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_size(8)
ax.set_title('会员的年龄分布')
#会员的男女比例
members.loc[:, '性别'] = pd.Categorical(members['性别'], categories=['男', '女'], ordered=True)
gender_counts = members['性别'].value_counts()
fig, ax = plt.subplots()
wedges, texts, autotexts = ax.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=90, wedgeprops=dict(width=0.3))
for text in texts:
    text.set_rotation(35)
for autotext in autotexts:
    autotext.set_color('black')
    autotext.set_size(8)
ax.set_title('会员的男女比例')
#男女消费金额比例
sales_type=df.groupby('性别')['单价'].sum()
plt.figure(figsize=(8, 8))
plt.pie(sales_type, labels=sales_type.index, autopct='%1.1f%%', startangle=140)
plt.title('男女消费金额比例')
plt.axis('equal')
#个性词云图
text = ' '.join(user_behavior_products['具体商品'].astype(str))
wordcloud = WordCloud(font_path = r'C:/WINDOWS/Fonts/SIMHEI.TTF',width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('热门商品')
plt.show()


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
plt.rcParams['font.sans-serif'] = ['SimHei']
user_behavior_products = pd.read_csv(r"E:\大三上\点宽杯\用户行为买了什么商品.csv")
# 统计不同消费行为的个数
payment_methods = user_behavior_products['行为'].value_counts()
# 绘制饼图
plt.figure(figsize=(8, 8))
plt.pie(payment_methods, labels=payment_methods.index, autopct='%1.1f%%', startangle=140)
plt.title('消费手段分布')
plt.axis('equal')
#用户行为分布漏斗模型
funnel_stages = ['浏览', '收藏', '加购', '购买']
funnel_counts = user_behavior_products['行为'].value_counts()
funnel_data = funnel_counts.loc[funnel_stages]
conversion_rates = funnel_data.div(funnel_data['浏览']) * 100
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 1)
ax.set_ylim(0, len(funnel_data) + 1)
ax.set_yticks(range(len(funnel_data)))
ax.set_yticklabels(funnel_data.index)
width = 0.8
heights = funnel_data.values
bottom_widths = heights / max(heights) * width
for i, (stage, count) in enumerate(funnel_data.items()):
    top_width = bottom_widths[i - 1] if i > 0 else width
    bottom_left = (1 - bottom_widths[i]) / 2
    bottom_right = (1 + bottom_widths[i]) / 2
    top_left = (1 - top_width) / 2
    top_right = (1 + top_width) / 2
    vertices = [(bottom_left, i + 1), (bottom_right, i + 1), (top_right, i), (top_left, i)]
    ax.add_patch(Polygon(vertices, closed=True, fill=True, alpha=0.7))
for i, (stage, count) in enumerate(funnel_data.items()):
    ax.text(-0.1, i + 0.5, stage, ha='right', va='center', fontsize=10)
    if i == 0:
        ax.text(1.1, i + 0.5, '100%', ha='left', va='center', fontsize=8, color='black')
    else:
        prev_count = funnel_data.iloc[i - 1]
        conversion_rate = (count / prev_count) * 100
        ax.text(1.1, i + 0.5, f'{conversion_rate:.1f}%', ha='left', va='center', fontsize=8, color='black')
ax.set_title('用户行为分布漏斗模型')
ax.set_xlabel('用户行为转化率')
ax.set_xticks([])
ax.set_yticks([])
plt.tight_layout()
plt.show()
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


def load_product_data(file_path):
    data = pd.read_csv(file_path)
    descriptions = data['具体商品'].astype(str)  # 确保所有数据都是字符串类型
    id_to_name = data.set_index('商品ID')['具体商品'].to_dict()
    return descriptions, id_to_name


# 根据商品ID查找商品名称
def get_product_name(product_id, id_to_name):
    return id_to_name.get(product_id, "您可能感兴趣的商品：")


# 计算基于内容的推荐
def recommend_based_on_content(descriptions, item_index, top_k=5):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(descriptions)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    sim_scores = list(enumerate(cosine_sim[item_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:top_k + 1]
    item_indices = [i[0] for i in sim_scores]
    return item_indices


# 读取用户行为数据
def load_user_behavior_data(file_path):
    data = pd.read_csv(file_path)
    data = data.dropna()

    # 定义行为到评分的映射
    behavior_to_rating = {
        '浏览': 0,
        '收藏': 1,
        '加购': 2,
        '购买': 3
    }
    data['评分'] = data['行为'].map(behavior_to_rating)
    data['时间'] = pd.to_datetime(data['时间'])
    data_sorted = data.sort_values(['用户ID', '商品ID', '时间'])
    data_latest = data_sorted.drop_duplicates(subset=['用户ID', '商品ID'], keep='last')

    return data_latest


# 计算基于用户的推荐
def recommend_based_on_users(data_latest, user_id, top_k=5):
    user_item_matrix = data_latest.pivot(index='用户ID', columns='商品ID', values='评分').fillna(0)
    user_item_sparse = csr_matrix(user_item_matrix.values)
    user_sim = cosine_similarity(user_item_sparse)

    user_index = user_item_matrix.index.get_loc(user_id)
    sim_scores = user_sim[user_index]
    sim_users = sorted(enumerate(sim_scores), key=lambda x: x[1], reverse=True)
    sim_users = sim_users[1:top_k + 1]

    recommended_items = set()
    for sim_user_idx, sim_score in sim_users:
        sim_user_items = set(user_item_matrix.columns[(user_item_matrix.iloc[sim_user_idx] > 0).to_numpy()])
        recommended_items.update(sim_user_items)

    user_items = set(user_item_matrix.columns[(user_item_matrix.loc[user_id] > 0).to_numpy()])
    recommended_items.difference_update(user_items)

    return list(recommended_items)


def hybrid_recommend(user_id, item_index, descriptions, data_latest, content_weight=0.5, user_weight=0.25,
                     item_weight=0.25):
    content_recs = recommend_based_on_content(descriptions, item_index, top_k=5)
    user_recs = recommend_based_on_users(data_latest, user_id, top_k=5)

    all_recs = content_recs + user_recs
    recs_series = pd.Series(all_recs)
    recs_counts = recs_series.value_counts()
    weighted_scores = pd.Series(index=recs_counts.index, dtype=float)
    for rec in content_recs:
        weighted_scores.at[rec] += recs_counts.get(rec, 0) * content_weight
    for rec in user_recs:
        weighted_scores.at[rec] += 1 * user_weight
    weighted_scores = weighted_scores.sort_values(ascending=False)
    return list(weighted_scores.index[:5])


# 主函数
def main(user_id):
    product_data_path = r"E:\大三上\点宽杯\商品（id类型具体商品成本单价库存）.csv"
    user_behavior_data_path = r"E:\大三上\点宽杯\用户行为商品时间备注.csv"
    descriptions, id_to_name = load_product_data(product_data_path)
    data_latest = load_user_behavior_data(user_behavior_data_path)

    # 基于内容的推荐
    item_index_to_recommend = 0  # 示例：推荐与第一个商品相似的商品
    recommended_items_content = recommend_based_on_content(descriptions, item_index_to_recommend)
    # 将商品ID转换为商品名称
    recommended_items_content_names = [get_product_name(item, id_to_name) for item in recommended_items_content]

    # 基于用户的推荐
    if user_id in data_latest['用户ID'].values:
        recommended_items_users = recommend_based_on_users(data_latest, user_id)
        recommended_items_users_names = [get_product_name(item, id_to_name) for item in recommended_items_users]

        # 混合推荐
        recommended_items_hybrid = hybrid_recommend(user_id, item_index_to_recommend, descriptions, data_latest)
        recommended_items_hybrid_names = [get_product_name(item, id_to_name) for item in recommended_items_hybrid]

        print(f"用户'{user_id}':", recommended_items_hybrid_names)
    else:
        print(f"用户ID '{user_id}' 不存在")


if __name__ == "__main__":
    user_id_to_recommend = input("请输入用户ID: ")
    main(user_id_to_recommend)
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'E:/大三上/点宽杯/用户行为买了什么商品.csv'
df = pd.read_csv(file_path)

# 筛选出“购买”和“加购”行为，并提取时间和单价
sales_df = df[df['行为'] == '购买'].copy()
sales_df['时间'] = pd.to_datetime(sales_df['时间'])

# 按天聚合销量和销售额
sales_df.set_index('时间', inplace=True)
sales_daily = sales_df.resample('D').agg({'单价': 'mean', '用户ID': 'count'})
sales_daily.columns = ['平均单价', '销量']
sales_daily['销售额'] = sales_daily['平均单价'] * sales_daily['销量']

# 检查数据的完整性，确保没有缺失值
sales_daily = sales_daily.dropna()

# 划分训练集和测试集（这里为了简化，我们直接使用全部数据作为训练集）
# 但在实际应用中，你应该保留一部分最近的数据作为测试集来验证模型的准确性
train_data = sales_daily['销量'].values

# 尝试拟合ARIMA模型
# 注意：这里我们选择了ARIMA(1, 1, 1)作为示例，但你需要根据数据的实际情况来调整参数
# 你也可以使用AIC、BIC等准则来选择最优的模型参数
model = ARIMA(train_data, order=(1, 1, 1))
model_fit = model.fit()

# 预测2024年第四季度的销量
# 假设当前时间是2023年底或2024年初，我们需要预测到2024年12月31日
start_date = sales_daily.index[-1] + timedelta(days=1)  # 最后一个日期之后的第一天
end_date = datetime(2024, 12, 31)
预测步数 = (结束日期 - 开始日期).天数 + 1
forecast_index = pd.date_range(start=start_date, periods=forecast_steps, freq='D')

预测 = 模型拟合.forecast(步骤=预测步数)
forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=['预测销量'])

# 计算预测的销售额（使用最后一段时间的平均单价作为预测单价）
last_avg_price = sales_daily['平均单价'].iloc[-1]
forecast_df['预测销售额'] = forecast_df['预测销量'] * last_avg_price

# 提取2024年第四季度的预测结果
q4_forecast = forecast_df[forecast_df.index.month >= 10]  # 这里包括10月、11月和12月，即第四季度

# 打印预测结果
print("2024年第四季度预测销量总和:", q4_forecast['预测销量'].sum())
print("2024年第四季度预测销售额总和:", q4_forecast['预测销售额'].sum())
# 可视化预测结果（可选）
plt.figure(figsize=(14, 7))
plt.rcParams['字体无衬线']='['思源黑体']
plt.plot(sales_daily.index, sales_daily['销量'], label='实际销量')
plt.plot(forecast_df.index, forecast_df['预测销量'], label='预测销量', color='red')
plt.xlabel('时间')
plt.ylabel('销量')
plt.title('销量预测')
plt.legend()
显示图表。



