_CMAP = {
    "帽": {"alias": "帽", "color": "#F7815D"},
    "领": {"alias": "领", "color": "#F9D26D"},
    "肩": {"alias": "肩", "color": "#F23434"},
    "袖片": {"alias": "袖片", "color": "#C4DBBE"},
    "袖口": {"alias": "袖口", "color": "#F0EDA8"},
    "衣身前中": {"alias": "衣身前中", "color": "#8CA740"},
    "衣身后中": {"alias": "衣身后中", "color": "#4087A7"},
    "衣身侧": {"alias": "衣身侧", "color": "#DF7D7E"},
    "底摆": {"alias": "底摆", "color": "#DACBBD"},
    "腰头": {"alias": "腰头", "color": "#DABDD1"},
    "裙前中": {"alias": "裙前中", "color": "#46B974"},
    "裙后中": {"alias": "裙后中", "color": "#6B68F5"},
    "裙侧": {"alias": "裙侧", "color": "#D37F50"},

    "橡筋": {"alias": "橡筋", "color": "#696969"},
    "木耳边": {"alias": "木耳边", "color": "#A8D4D2"},
    "袖笼拼条": {"alias": "袖笼拼条", "color": "#696969"},
    "荷叶边": {"alias": "荷叶边", "color": "#A8D4D2"},
    "绑带": {"alias": "绑带", "color": "#696969"}
}

_CMAP_LINE = {
    "领窝线": {"alias": "领窝线", "color": "#8000FF"},
    "袖笼弧线": {"alias": "袖笼弧线", "color": "#00B5EB"},      # interfaces between sleeve panels and bodice panels (belongs to bodice panels)
    "袖山弧线": {"alias": "袖山弧线", "color": "#00B5EB"},      # interfaces between sleeve panels and bodice panels (belongs to sleeve panels)
    "腰线": {"alias": "腰线", "color": "#80FFB4"},
    "袖口线": {"alias": "袖口线", "color": "#FFB360"},
    "底摆线": {"alias": "底摆线", "color": "#FF0000"},

    "省": {"alias": "省道", "color": "#FF3333"},
    "褶": {"alias": "褶", "color": "#33FF33"},
}

_PANEL_CLS = [
    '帽', '领', '肩', '袖片', '袖口', '衣身前中', '衣身后中', '衣身侧', '底摆', '腰头',
    '裙前中', '裙后中', '裙侧', '橡筋', '木耳边', '袖笼拼条', '荷叶边', '绑带']

_PANEL_CLS_ENG = [
    'hat', 'collar', 'shoulder', 'sleeve', 'cuff', 'body front', 'body back', 'dbody side', 'hem', 'waist',
    'skirt front', 'skirt back', 'skirt side', '', '', '', '', '']

data_fields_dict = {
    "surf_ncs": {
        "title": "Geometry Images",
        "len": 3,
    },
    "surf_wcs": {
        "title": "Geometry_Wcs Images",
        "len": 3,
    },
    "surf_uv_ncs": {
        "title": "UV Images",
        "len": 2,
    },
    "surf_normals": {
        "title": "Normals Images",
        "len": 3,
    },
    "surf_mask": {
        "title": "Mask Images",
        "len": 1,
    },
    "latent64": {
        "title": None,
        "len": 64,
    },
    "bbox3d": {
        "title": None,
        "len": 6,
    },
    "scale3d": {
        "title": None,
        "len": 3,
    },
    "bbox2d": {
        "title": None,
        "len": 4,
    },
    "scale2d": {
        "title": None,
        "len": 2,
    }
}