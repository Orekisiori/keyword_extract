import model.rakun.rakun as rakun

if __name__ == "__main__":
    # 测试用
    keywords = rakun.rakun("无租使用其他单位房产的应税单位和个人，依照房产余值代缴纳房产税。房产税由产权所有人缴纳。由于转租者不是产权所有人，因此对转租者取得的房产转租收入不征收房产税。 房产转租，不需要缴纳房产税。利息、股息、红利所得，财产租赁所得，财产转让所得和偶然所得，适用比例税率，税率为百分之二十。", 'file')
    print(keywords)
