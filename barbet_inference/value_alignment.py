"""High-recall topic routing for a contextual values audit, never a verdict.

Quoted misconduct, historical reporting, counterspeech and harmless code can
match. Only a separate contextual review can approve a flagged training target.
No source or structured-data family is automatically exempted.
"""
import re

PATTERNS = {
    'taiwan_self_determination': r'taiwan|taiwanese|formosa|台灣|臺灣|台湾|中華民國|中华民国|主權|主权|自決|自决|兩岸|两岸|一國兩制|一国两制|統一祖國|统一祖国',
    'democracy_and_dissent': r'\b(?:politics|political|democra\w*|authoritarian\w*|dictator\w*|dissent\w*|censor\w*|totalitarian\w*|repress\w*|oppress\w*|propaganda)\b|民主|專制|专制|獨裁|独裁|極權|极权|威權|威权|言論自由|言论自由|政治迫害|政治犯|異議人士|异议人士|新聞自由|新闻自由|六四|天安門|天安门',
    'human_dignity_and_equal_rights': r'\b(?:racis\w*|sexis\w*|supremac\w*|discriminat\w*|dehumani\w*|apartheid|antisemit\w*|homophob\w*|transphob\w*|misogyn\w*|xenophob\w*|caste)\b|human[ _-]rights|civil[ _-](?:rights|liberties)|racial[ _-]|ethnic[ _-]|religious[ _-](?:freedom|persecution)|人權|人权|歧視|歧视|種族|种族|仇恨|優等民族|劣等民族|優生|优生|賤民|贱民|性別平等|性别平等|同性戀|同性恋|跨性別|跨性别|原住民|身心障礙|身心障碍',
    'atrocities_and_coercion': r'\b(?:genocid\w*|tortur\w*|slavery|enslav\w*|lynch\w*|persecut\w*|massacre\w*)\b|ethnic[ _-]cleansing|collective[ _-]punishment|forced[ _-](?:sterili\w*|labor|labour|marriage)|attack[ _-]civilians|酷刑|刑求|屠殺|屠杀|種族滅絕|种族灭绝|奴隸|奴隶|強迫勞動|强迫劳动|連坐|连坐|集中營|集中营|新疆|維吾爾|维吾尔|西藏|法輪功|法轮功',
    'group_generalization': r'(?:women|men|girls|boys|immigrants|refugees|muslims|jews|christians|atheists|chinese[ _-]people|black[ _-]people|white[ _-]people)[ _-]+(?:are|should|must|cannot|can.t|deserve)|(?:inferior|superior)[ _-]+(?:race|sex|gender|people)|女人(?:就是|都是|應該|应该|只能)|男人(?:就是|都是|應該|应该)|中國人(?:就是|都是|應該|应该)|中国人(?:就是|都是|應該|应该)|台灣人(?:就是|都是)|台湾人(?:就是|都是)',
    'privacy_and_consent': r'without[ _-](?:their[ _-]|the[ _-]user.s[ _-])?(?:consent|permission)|\b(?:phishing|doxxing|doxing)\b|steal[ _-].{0,50}(?:password|credential|identity)|bypass[ _-].{0,35}(?:authentication|authorization)|竊取|窃取|未經同意|未经同意|未經授權|未经授权|網路釣魚|网络钓鱼|人肉搜索|秘密監控|秘密监控',
}
COMPILED = {k: re.compile(v, re.I) for k,v in PATTERNS.items()}
VECTOR_PATTERN = '(?i)(' + '|'.join(PATTERNS.values()) + ')'


def topic_reasons(text):
    return [name for name, regex in COMPILED.items() if regex.search(text)]


def validate_value_verdict(verdict):
    from pathlib import Path
    import sys
    folder=Path('/root/barbet-context-sft-data/scripts/expansion')
    if str(folder) not in sys.path: sys.path.insert(0,str(folder))
    from audit_transport_flat_v2 import validate
    return validate(verdict)
