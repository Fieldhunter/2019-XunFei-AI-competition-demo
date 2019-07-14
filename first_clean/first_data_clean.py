import pandas as pd
import json


# 删除部分字段
def del_features(data):
	del_list = ["sid", "ver", "province", "idfamd5", "make", "os"]
	for i in del_list:
		del data[i]


# 将nginxtime转换为相对时间，再转化为秒
def nginxtime(data):
	new_nginxtime = (data["nginxtime"]-data["nginxtime"].min())/1000
	data["nginxtime"] = new_nginxtime


def dvctype(data):
	new_dvctype = []
	for i in data["dvctype"]:
		if i == 3:
			i = 1
		new_dvctype.append(i)
	data["dvctype"] = new_dvctype


def orientation(data):
	new_orientation = []
	for i in data["orientation"]:
		if i == 90:
			i = 2
		new_orientation.append(i)
	data["orientation"] = new_orientation


# 形成新字段像素值，resolution_ratio
def h_w_ppi(data):
	new_h = data["h"]+0.1
	new_w = data["w"]+0.1
	new_ppi = data["ppi"]+0.1
	resolution_ratio = new_h * new_w * new_ppi

	del data["h"]
	del data["w"]
	del data["ppi"]
	data["resolution_ratio"] = resolution_ratio


def apptype(data1, data2):
	global json_dict

	def process(data):
		new_apptype = []
		for i in data["apptype"]:
			try: 
				new_apptype.append(apptype_set.index(i))
			except:
				new_apptype.append(len(apptype_set))
		data["apptype"] = new_apptype

	apptype_set = list(set(data1["apptype"]) | set(data2["apptype"]))
	json_dict["apptype"] = apptype_set

	process(data1)
	process(data2)


def carrier(data1, data2):
	global json_dict

	def process(data):
		new_carrier = []
		for i in data["carrier"]:
			new_carrier.append(carrier_set.index(i))
		data["carrier"] = new_carrier

	carrier_set = list(set(data1["carrier"]) | set(data2["carrier"]))
	json_dict["carrier"] = carrier_set

	process(data1)
	process(data2)


def lan(data1, data2):
	global json_dict

	def process_1(data):
		new_lan = []
		for i in data["lan"]:
			try:
				for j in i:
					if j in lan_del_string:
						i = i.replace(j,"")

				i = i.lower()
			except:
				pass
			finally:
				new_lan.append(i)

		return new_lan

	def process_2(data, new_lan):
		for n, i in enumerate(new_lan):
			try:
				new_lan[n] = lan_set.index(i)
			except:
				new_lan[n] = len(lan_set)
		data["lan"] = new_lan

	lan_del_string = ["_", "-"]

	new_lan_1 = process_1(data1)
	new_lan_2 = process_1(data2)

	lan_set = list(set(new_lan_1) | set(new_lan_2))
	json_dict["lan"] = lan_set

	process_2(data1, new_lan_1)
	process_2(data2, new_lan_2)


def model(data):
	model_del_string = [" ", "_", ",", "+", "/", "-", "%", "(", ")", "."]
	new_model = []
	for i in data["model"]:
		try:
			for j in i:
				if j in model_del_string:
					i = i.replace(j,"")

			i = i.lower()
			if "huaweihuawei" in i:
				i = i.replace("huaweihuawei","huawei")
			if "xiaomixiaomi" in i:
				i = i.replace("xiaomixiaomi","xiaomi")
		except:
			pass
		finally:
			new_model.append(i)

	data["model"] = new_model


def osv(data):
	new_osv = []
	for i in data["osv"]:
		try:
			if "," in i:
				i = i.replace(",", ".")
			if " 十核2.0G_HD" in i:
				i = i.replace(" 十核2.0G_HD", "")

			process = i.split(".")
			while process[-1] == "0":
				del process[-1]
			i = ".".join(process)
		except:
			pass
		finally:
			new_osv.append(i)
	
	data["osv"] = new_osv


# 将连续值，需要one-hot的特征，需要embedding的特征分开排放
def change_col_index(data):
	col_index = ["label", "nginxtime", "ip", "resolution_ratio", "apptype", \
				"dvctype", "ntt", "carrier", "orientation", "lan", "pkgname", \
				"adunitshowid", "mediashowid", "city", "reqrealip", "adidmd5", \
				"imeimd5","openudidmd5", "macmd5", "model", "osv"]
	if data is data2:
		del col_index[0]

	return data[col_index]


# 第一轮清洗
def first_data_clean(data1, data2):
	apptype(data1, data2)
	carrier(data1, data2)
	lan(data1, data2)

	for i in [data1, data2]:
		del_features(i)
		nginxtime(i)
		dvctype(i)
		orientation(i)
		h_w_ppi(i)
		model(i)
		osv(i)


# one-hot_json.json中存储所有需要one-hot特征的索引，不要轻易改动
if __name__ == "__main__":
	data1_path = "~/桌面/移动广告/data/train.csv"
	data2_path = "~/桌面/移动广告/data/test.csv"
	data1_to_csv_path = "~/桌面/移动广告/data/first_train_clean.csv"
	data2_to_csv_path = "~/桌面/移动广告/data/first_test_clean.csv"
	json_path = "one-hot_json.json"

	data1 = pd.read_csv(data1_path)
	data2 = pd.read_csv(data2_path)
	json_dict = {}

	first_data_clean(data1, data2)
	data1 = change_col_index(data1)
	data2 = change_col_index(data2)

	data1.to_csv(data1_to_csv_path, index=False)
	data2.to_csv(data2_to_csv_path, index=False)
	with open(json_path,"w") as f:
		json.dump(json_dict, f)
