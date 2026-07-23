from __future__ import annotations

import csv
from functools import lru_cache
from pathlib import Path

from .data import TEAM_NAME_ALIASES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUSTOM_TEAM_NAMES_PATH = PROJECT_ROOT / "data/team_name_overrides.csv"

TEAM_NAME_ZH = {
    "Afghanistan": "阿富汗",
    "Albania": "阿尔巴尼亚",
    "Algeria": "阿尔及利亚",
    "American Samoa": "美属萨摩亚",
    "Andorra": "安道尔",
    "Angola": "安哥拉",
    "Anguilla": "安圭拉",
    "Antigua and Barbuda": "安提瓜和巴布达",
    "Argentina": "阿根廷",
    "Armenia": "亚美尼亚",
    "Arsenal": "阿森纳",
    "Aruba": "阿鲁巴",
    "Australia": "澳大利亚",
    "Austria": "奥地利",
    "Azerbaijan": "阿塞拜疆",
    "Bahamas": "巴哈马",
    "Bahrain": "巴林",
    "Bangladesh": "孟加拉国",
    "Barbados": "巴巴多斯",
    "Belarus": "白俄罗斯",
    "Belgium": "比利时",
    "Belize": "伯利兹",
    "Benin": "贝宁",
    "Bermuda": "百慕大",
    "Bhutan": "不丹",
    "Bolivia": "玻利维亚",
    "Bosnia and Herzegovina": "波黑",
    "Botswana": "博茨瓦纳",
    "Brazil": "巴西",
    "British Virgin Islands": "英属维尔京群岛",
    "Brunei": "文莱",
    "Bulgaria": "保加利亚",
    "Burkina Faso": "布基纳法索",
    "Burundi": "布隆迪",
    "Cabo Verde": "佛得角",
    "Cambodia": "柬埔寨",
    "Cameroon": "喀麦隆",
    "Canada": "加拿大",
    "Cape Verde": "佛得角",
    "Cayman Islands": "开曼群岛",
    "Central African Republic": "中非共和国",
    "Chad": "乍得",
    "Chelsea": "切尔西",
    "Chile": "智利",
    "China PR": "中国",
    "Colombia": "哥伦比亚",
    "Comoros": "科摩罗",
    "Congo": "刚果共和国",
    "Cook Islands": "库克群岛",
    "Costa Rica": "哥斯达黎加",
    "Croatia": "克罗地亚",
    "Cuba": "古巴",
    "Curacao": "库拉索",
    "Curaçao": "库拉索",
    "Cyprus": "塞浦路斯",
    "Czech Republic": "捷克",
    "Czechia": "捷克",
    "DR Congo": "民主刚果",
    "Denmark": "丹麦",
    "Djibouti": "吉布提",
    "Dominica": "多米尼克",
    "Dominican Republic": "多米尼加共和国",
    "Ecuador": "厄瓜多尔",
    "Egypt": "埃及",
    "El Salvador": "萨尔瓦多",
    "England": "英格兰",
    "Equatorial Guinea": "赤道几内亚",
    "Estonia": "爱沙尼亚",
    "Eswatini": "斯威士兰",
    "Ethiopia": "埃塞俄比亚",
    "Faroe Islands": "法罗群岛",
    "Fiji": "斐济",
    "Finland": "芬兰",
    "France": "法国",
    "Gabon": "加蓬",
    "Gambia": "冈比亚",
    "Georgia": "格鲁吉亚",
    "Germany": "德国",
    "Ghana": "加纳",
    "Gibraltar": "直布罗陀",
    "Greece": "希腊",
    "Grenada": "格林纳达",
    "Guam": "关岛",
    "Guatemala": "危地马拉",
    "Guinea": "几内亚",
    "Guinea-Bissau": "几内亚比绍",
    "Guyana": "圭亚那",
    "Haiti": "海地",
    "Honduras": "洪都拉斯",
    "Hong Kong": "中国香港",
    "Hungary": "匈牙利",
    "Iceland": "冰岛",
    "India": "印度",
    "Indonesia": "印度尼西亚",
    "Iran": "伊朗",
    "Iraq": "伊拉克",
    "Israel": "以色列",
    "Italy": "意大利",
    "Ivory Coast": "科特迪瓦",
    "Jamaica": "牙买加",
    "Japan": "日本",
    "Jordan": "约旦",
    "Kazakhstan": "哈萨克斯坦",
    "Kenya": "肯尼亚",
    "Kosovo": "科索沃",
    "Kuwait": "科威特",
    "Kyrgyzstan": "吉尔吉斯斯坦",
    "Laos": "老挝",
    "Latvia": "拉脱维亚",
    "Lebanon": "黎巴嫩",
    "Lesotho": "莱索托",
    "Liberia": "利比里亚",
    "Libya": "利比亚",
    "Liechtenstein": "列支敦士登",
    "Lithuania": "立陶宛",
    "Liverpool": "利物浦",
    "Luxembourg": "卢森堡",
    "Macau": "中国澳门",
    "Madagascar": "马达加斯加",
    "Malawi": "马拉维",
    "Malaysia": "马来西亚",
    "Maldives": "马尔代夫",
    "Mali": "马里",
    "Malta": "马耳他",
    "Manchester City": "曼城",
    "Mauritania": "毛里塔尼亚",
    "Mauritius": "毛里求斯",
    "Mexico": "墨西哥",
    "Moldova": "摩尔多瓦",
    "Mongolia": "蒙古",
    "Montenegro": "黑山",
    "Montserrat": "蒙特塞拉特",
    "Morocco": "摩洛哥",
    "Mozambique": "莫桑比克",
    "Myanmar": "缅甸",
    "Namibia": "纳米比亚",
    "Nepal": "尼泊尔",
    "Netherlands": "荷兰",
    "New Caledonia": "新喀里多尼亚",
    "New Zealand": "新西兰",
    "Nicaragua": "尼加拉瓜",
    "Niger": "尼日尔",
    "Nigeria": "尼日利亚",
    "North Korea": "朝鲜",
    "North Macedonia": "北马其顿",
    "Northern Ireland": "北爱尔兰",
    "Norway": "挪威",
    "Oman": "阿曼",
    "Pakistan": "巴基斯坦",
    "Palestine": "巴勒斯坦",
    "Panama": "巴拿马",
    "Papua New Guinea": "巴布亚新几内亚",
    "Paraguay": "巴拉圭",
    "Peru": "秘鲁",
    "Philippines": "菲律宾",
    "Poland": "波兰",
    "Portugal": "葡萄牙",
    "Puerto Rico": "波多黎各",
    "Qatar": "卡塔尔",
    "Republic of Ireland": "爱尔兰",
    "Romania": "罗马尼亚",
    "Rwanda": "卢旺达",
    "Saint Kitts and Nevis": "圣基茨和尼维斯",
    "Saint Lucia": "圣卢西亚",
    "Saint Vincent and the Grenadines": "圣文森特和格林纳丁斯",
    "Samoa": "萨摩亚",
    "San Marino": "圣马力诺",
    "Saudi Arabia": "沙特阿拉伯",
    "Scotland": "苏格兰",
    "Senegal": "塞内加尔",
    "Serbia": "塞尔维亚",
    "Seychelles": "塞舌尔",
    "Sierra Leone": "塞拉利昂",
    "Singapore": "新加坡",
    "Slovakia": "斯洛伐克",
    "Slovenia": "斯洛文尼亚",
    "Solomon Islands": "所罗门群岛",
    "Somalia": "索马里",
    "South Africa": "南非",
    "South Korea": "韩国",
    "South Sudan": "南苏丹",
    "Spain": "西班牙",
    "Sri Lanka": "斯里兰卡",
    "Sudan": "苏丹",
    "Suriname": "苏里南",
    "Sweden": "瑞典",
    "Switzerland": "瑞士",
    "Syria": "叙利亚",
    "São Tomé and Príncipe": "圣多美和普林西比",
    "Tahiti": "塔希提",
    "Taiwan": "中国台北",
    "Tajikistan": "塔吉克斯坦",
    "Tanzania": "坦桑尼亚",
    "Thailand": "泰国",
    "Timor-Leste": "东帝汶",
    "Togo": "多哥",
    "Tonga": "汤加",
    "Trinidad and Tobago": "特立尼达和多巴哥",
    "Tunisia": "突尼斯",
    "Turkey": "土耳其",
    "Turkmenistan": "土库曼斯坦",
    "Turks and Caicos Islands": "特克斯和凯科斯群岛",
    "Uganda": "乌干达",
    "Ukraine": "乌克兰",
    "United Arab Emirates": "阿联酋",
    "United States": "美国",
    "United States Virgin Islands": "美属维尔京群岛",
    "Uruguay": "乌拉圭",
    "Uzbekistan": "乌兹别克斯坦",
    "Vanuatu": "瓦努阿图",
    "Venezuela": "委内瑞拉",
    "Vietnam": "越南",
    "Wales": "威尔士",
    "Beijing Guoan": "北京国安",
    "Changchun Yatai": "长春亚泰",
    "Chengdu Rongcheng": "成都蓉城",
    "Chongqing Tongliang Long": "重庆铜梁龙",
    "Dalian Yingbo": "大连英博",
    "Eritrea": "厄立特里亚",
    "Henan": "河南队",
    "Liaoning Tieren": "辽宁铁人",
    "Meizhou Hakka": "梅州客家",
    "Qingdao Hainiu": "青岛海牛",
    "Qingdao West Coast": "青岛西海岸",
    "Russia": "俄罗斯",
    "Shandong Taishan": "山东泰山",
    "Shanghai Port": "上海海港",
    "Shanghai Shenhua": "上海申花",
    "Shenzhen Peng City": "深圳新鹏城",
    "Tianjin Jinmen Tiger": "天津津门虎",
    "Wuhan Three Towns": "武汉三镇",
    "Yunnan Yukun": "云南玉昆",
    "Zhejiang Professional": "浙江队",
    "Yemen": "也门",
    "Zambia": "赞比亚",
    "Zimbabwe": "津巴布韦",
}


@lru_cache(maxsize=1)
def custom_team_names() -> dict[str, str]:
    if not CUSTOM_TEAM_NAMES_PATH.exists():
        return {}
    with CUSTOM_TEAM_NAMES_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return {
            str(row.get("team", "")).strip(): str(row.get("zh", "")).strip()
            for row in reader
            if str(row.get("team", "")).strip() and str(row.get("zh", "")).strip()
        }


def all_team_names_zh() -> dict[str, str]:
    names = TEAM_NAME_ZH.copy()
    names.update(custom_team_names())
    for alias in TEAM_NAME_ALIASES:
        names.pop(alias, None)
    return names


def save_custom_team_name(team: str, zh: str) -> None:
    team = team.strip()
    zh = zh.strip()
    if not team or not zh:
        return
    names = custom_team_names().copy()
    names[team] = zh
    CUSTOM_TEAM_NAMES_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CUSTOM_TEAM_NAMES_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["team", "zh"])
        writer.writeheader()
        for key in sorted(names):
            writer.writerow({"team": key, "zh": names[key]})
    custom_team_names.cache_clear()


def team_name_zh(team: str) -> str:
    return all_team_names_zh().get(team, team)


def team_display_name(team: str) -> str:
    zh = team_name_zh(team)
    if zh == team:
        return team
    return f"{zh}（{team}）"
