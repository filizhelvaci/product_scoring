###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.width', 170)


df = pd.read_csv("dataset/df_sub.csv")
df.head()
df.info()

# reviewerID - Kullanıcı ID'si
# asin - Ürün ID'si
# reviewerName - Kullanıcı Adı
# helpful - Faydalı yorum derecesi Örn: 2/3
# reviewText - Kullanıcının yazdığı inceleme metni
# overall - Ürün rating'i
# summary - inceleme özeti
# unixReviewTime - İnceleme zamanı (unix time)
# reviewTime - İnceleme zamanı (raw)

###################################################
# Bir ürünün rating'ini güncel yorumlara göre hesaplamak ve eski rating ile kıyaslamak
###################################################

#Veri setinde tek ürünmü var onun kontrolünü yapıyoruz.
df["asin"].value_counts()
df.shape

###################################################
# Ürünün ortalama puanı
###################################################

#Ürünün aldığı ortalama rating :  ~4.58
df["overall"].mean()

###################################################
# Tarihe göre ağırlıklı puan ortalaması hesaplamak
###################################################

#Ürün için yapılan güncel ratingleri yakalamak için tarihleri kontrol ediyoruz.
#reviewTime değişkenini to_datetime ile değişken türünü değiştirme.
df["reviewTime"]=pd.to_datetime(df["reviewTime"])

#Veri setindeki en son tarihi bulduk:
current_date=df["reviewTime"].max()
df["reviewTime"].sort_values(ascending=False).head(20)

#Son tarihten diğer tarihler çıkarılarak ratingler için kaçar gün geçtiği hesaplanarak yeni bir değişken olarak tutulur
df["days"]=(current_date - df["reviewTime"]).dt.days


###### PDF'teki örneğe göre : (4.60)
a=df["days"].quantile(0.25)
b=df["days"].quantile(0.50)
c=df["days"].quantile(0.75)

###################################################
# Önceki maddeden gelen a,b,c değerlerine göre ağırlıklı puanı hesaplıyoruz
###################################################

df.loc[df["days"]<=a, "overall"].mean() * 30/100 + \
    df.loc[(df["days"]>a)&(df["days"]<=b),"overall"].mean()*26/100 + \
    df.loc[(df["days"]>b)&(df["days"]<=c),"overall"].mean()*24/100 + \
    df.loc[(df["days"]>c),"overall"].mean()*20/100

#yeni ratingleri yakalamak için verideki en eski puan verilen günü buluyoruz.Buna göre gün sayılarına göre ağırlıklandırıyoruz.
df["days"].max() # 774
df["days"].min() # 0

# Başka bir açııdan bakmak istersek:

#Zamana göre rating dağılımını yaptığımızda : 4.64
df.loc[df["days"]<=90, "overall"].mean() * 30/100 + \
    df.loc[(df["days"]>90)&(df["days"]<=180),"overall"].mean()*26/100 + \
    df.loc[(df["days"]>180)&(df["days"]<=360),"overall"].mean()*24/100 + \
    df.loc[(df["days"]>360),"overall"].mean()*20/100

# Eski ratingimiz 4.58 di yeni ratingimiz 4.64 demekki son zamnalarda müşteriler bu üründe yada sağlanan hizmette fayda görmüşler.
# Yapılan yorumlar ayrıntılı incelenerek ürün iyileştirmesine tedarikçi firma sorunlarının giderilmesine, y
# ada kargo firmalarıyla yaşanan sorunların çözümlenmesine odaklanılabilir.
# Son 3 ay ratingi: 4.68
# Son 3-6 ay arası rating: 4.71
# Son 6-12 ay arası rating: 4.61
# 1 sene öncesi rating ortalamsı: 4.47

# Bu sıralamada User Kalitesine göre bir ağırlıklandırma yapılmamıştır.
# Eğer elimizde User kalitesini ölçebileceğiimiz metrikler olursa onlarıda kullanarak farklı bir product rating yapılabilir.


###################################################
# Product tanıtım sayfasında görüntülenecek ilk 20 yorumu belirleme:
###################################################

###################################################
# Helpful değişkeni içerisinden 3 değişken türetmek. 1: helpful_yes, 2: helpful_no,  3: total_vote
###################################################
df['helpful_yes'] = df[['helpful']].applymap(lambda x : x.split(', ')[0].strip('[')).astype(int)
df['total_vote'] = df[['helpful']].applymap(lambda x : x.split(', ')[1].strip(']')).astype(int)
df["helpful_no"]=df["total_vote"]-df["helpful_yes"]

###################################################
# score_pos_neg_diff'a göre skorlar oluşturunuz ve df_sub içerisinde score_pos_neg_diff ismiyle kaydediyoruz.
###################################################

df["score_pos_neg_diff"]=df["helpful_yes"]-df["helpful_no"]

df["helpful_yes"].sum()
df["helpful_no"].sum()

df.sort_values("score_pos_neg_diff", ascending=False)

###################################################
# score_average_rating'a göre skorlar oluşturuyoruz ve df_sub içerisinde score_average_rating ismiyle kaydediyoruz
###################################################

df["score_average_rating"]=df["helpful_yes"] / (df["helpful_yes"]+df["helpful_no"])

df.sort_values("score_average_rating", ascending=False)

##################################################
# wilson_lower_bound'a göre skorlar oluşturuyoruz ve df_sub içerisinde wilson_lower_bound ismiyle kaydediyoruz.
###################################################

def wilson_lower_bound(pos, neg, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not: Eğer skorlar 1-5 arasıdaysa 1-3 down, 4-5 up olarak işaretlenir ve bernoulli'ye uygun hale getirilir.

    Parameters
    ----------
    pos: int
        pozitif yorum sayısı
    neg: int
        negatif yorum sayısı
    confidence: float
        güven aralığı

    Returns
    -------
    wilson score: float

    """
    n = pos + neg
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * pos / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["WLB_SCORE"]=""

for i in df.index:
    df["WLB_SCORE"][i] = wilson_lower_bound(df["helpful_yes"][i],df["helpful_no"][i])



##################################################
# Ürün sayfasında gösterilecek 20 yorumu belirleyip, sonuçları yorumluyoruz.
###################################################
# En akla yatkın sıralama şekli Wilsoon_lower_bound olduğu için bunu kullanarak sıraladık çünkü burada sıralama yaparken
# Helpful_yes ile Helpful_no arasındaki fark fazla olsada oransal olarak fazla olanında değerini koruyacak şekilde problemi çözdü.
# Ayrıca Helpful_yes sayısının değerinide korudu.Yani Yani örneğin: Helpful_no değerleri olmayan helpful_yes=50 değerini helpful_yes=7 değerine karşı korudu.
# Bundan önceki iki skorlamada karşımıza çıkan problemleri çözmüş oldu.
# User kalitesi ile ilgili veri setinde bilgiler olduğunda bunlardanda değişkenler oluşturularak
# sıralamada user kaliteside bir metrik olarak kullanılabilir.
# Bu sıralamada yapılan yorumların olumlu yada olumsuz olmasına bakılmadı. Yapılan yorumları müşterilerin
# faydalı bulup bulmadığına bakılarak sıralama yapıldı.

df.sort_values("WLB_SCORE", ascending=False).head(20)
