"""Optional deterministic *candidate* routing from causal inputs only.

Call after an application's more specific, reviewed router returns no skill.
No targets, dataset IDs, source labels, scores, or future messages are accepted.
Selection is applicability evidence, not proof of correct answers or execution.
English/Traditional Chinese lexical coverage is deliberately incomplete.
"""
import re
VERSION = 'causal-procedure-extension/2'

def has(pattern, text):
    return bool(re.search(pattern, text, re.I))

def _request(model_input):
    if not isinstance(model_input, dict) or set(model_input) != {'messages', 'tools'}:
        raise ValueError('Only causal messages and tools are accepted')
    messages, tools = model_input['messages'], model_input['tools']
    if not isinstance(messages, list) or not isinstance(tools, list):
        raise ValueError('messages/tools must be lists')
    if any(not isinstance(m, dict) or m.get('role') not in {'system','user','assistant','tool'}
           or not isinstance(m.get('content'), str) for m in messages):
        raise ValueError('Invalid causal messages')
    if not messages or messages[-1]['role'] != 'user':
        return None
    text = messages[-1]['content'].strip()
    if not text or re.fullmatch(r"(?:thanks?|thank you|no thanks|no thank you|no|okay|ok|great|bye|goodbye|hello|hi|謝謝|你好|不用)[.!\s]*", text, re.I):
        return None
    if any('[BARBET_SKILL_CONTEXT_V1]' in m['content'] for m in messages if m['role']=='system'):
        return None
    return messages, tools, text


def select_factual_property(model_input):
    """Narrow hook before a broad concept fallback, after specific task routes.

    Recognizes single-clause property questions, not arbitrary named entities.
    Abstains on tools, documents, quotations, operations and live-state wording.
    It must not replace an application's source/code/translation/math router.
    """
    request = _request(model_input)
    if request is None:
        return None
    messages, tools, text = request
    if any(has(r'參考資料|閱讀資料|資料表|根據以下|\b(?:according to|passage|document|source text)\b', message['content'])
           for message in messages):
        return None
    if tools or has(r'[\n\r:：;；"\'“”「」`]|根據|依據|參考資料|閱讀資料|資料表|段落|文件|翻譯|改寫|摘要|計算|求解|比較|程式|演算法|而且|以及|目前|現在|最新|今天|即時|此刻|\b(?:current|currently|latest|today|now|live|translate|rewrite|summarize|calculate|compute|solve|compare|code|function|program|given|following|and|why|explain)\b', text):
        return None
    # Entity digits are labels, not arithmetic operands. No answer values occur
    # in these rules; changing a target or its source must not change selection.
    attribute = r'(?:建築高度|高度|海拔高度|海拔|樓層數|樓層|軌距|全長|所在地|地址|面積|首都|幣別|貨幣名稱|開通年份|啟用年份|成立年份|出生日期)'
    entity = r'[^，,。.!?？;；\n]{1,80}'
    question = text.rstrip('？?。.! ').strip()
    chinese = (
        rf'(?:請問[，, ]*)?(?:什麼是|何謂){entity}的{attribute}',
        rf'(?:請問[，, ]*)?{entity}的{attribute}(?:是多少|是什麼|為多少|為何|是多少公尺|是幾公尺|是幾米|有幾層|在哪裡)',
        rf'(?:請問[，, ]*)?{entity}的{attribute}(?:是|為)[+-]?\d+(?:\.\d+)?(?:公尺|公分|公釐|米|層|年|公里|平方公里)?嗎',
        rf'(?:請說明|說明|請告訴我|告訴我){entity}的{attribute}',
        rf'(?:請問[，, ]*)?{entity}(?:有幾層|多高|位於哪裡)',
    )
    english = (
        r'(?:please tell me )?what (?:is|was) (?:the )?(?:height|elevation|altitude|gauge|length|area|location|capital|currency|opening year|founding year) of .{1,80}',
        r'how (?:tall|high|long) (?:is|was|are|were) .{1,80}',
        r'how many (?:floors|storeys|stories) (?:does|did) .{1,80} have',
    )
    if any(re.fullmatch(pattern, question, re.I) for pattern in chinese + english):
        return {'skill': 'factual-question-resolution',
                'rule': 'explicit_entity_property_question',
                'input_turn': len(messages) - 1}
    return None


def select(model_input):
    request = _request(model_input)
    if request is None:
        return None
    messages, tools, text = request
    def pick(name, rule):
        return {'skill': name, 'rule': rule, 'input_turn': len(messages)-1}
    if tools:
        if any(not isinstance(t,dict) or not isinstance(t.get('function'),dict)
               or not t['function'].get('name') for t in tools):
            return None
        return pick('tool-capability-triage', 'visible_registry_and_unresolved_request')
    # Supplied data and explicit operations precede broad question templates.
    numeric = has(r'\d', text)
    supplied = has(r"[:：\n]|[\[\]\"'“”]|\bgiven\b|\bfollowing\b", text)
    if has(r'\b(random|fake|lorem ipsum|placeholder|simulate|simulation|draw|drawing|dice|password|alphanumeric|payment card numbers)\b|\d+.character',text) and has(r'\b(generate|produce|create|need|want|draw|drawing|simulate|rolling|cards?|password|characters?|string)\b',text) and not has(r'\b(code|function|program|algorithm|explain)\b',text):
        return pick('constrained-sample-generation','explicit_sample_or_simulation_constraints')
    if has(r'\b(anagrams?|palindrom\w*|similarity between)\b',text) and supplied and not has(r'\b(vectors?|cosine)\b',text):
        return pick('string-property-analysis','explicit_string_property_and_values')
    if has(r'\b(validat\w*|verify|validity|check|filter)\b',text) and has(r'\b(phone|number|email|domain|profanity|tweet|format)\b',text) and not has(r'\b(code|function|program|algorithm)\b',text):
        return pick('validation-scope-check','explicit_validation_scope')
    if has(r'\b(screenshot|thumbnail|backup|blur|grayscale|rotate|resize|image|face|static analysis|dynamic analysis)\b',text) and has(r'\b(take|generate|create|convert|blur|resize|need|perform|rotate)\b',text):
        return pick('artifact-operation-contract','explicit_artifact_operation')
    numeric_collection=has(r'\[[^\]]*\d|[:\n]\s*[+-]?\d+(?:\s*,\s*[+-]?\d+)+',text)
    if numeric and has(r'\b(largest|smallest|top\s+\d+|sort|sorted|ranking)\b|排序|最大|最小', text) and numeric_collection:
        return pick('ordered-top-k', 'explicit_order_statistic_with_values')
    if numeric and has(r'\b(mean|median|mode|variance|standard deviation|average|frequency)\b|平均數|中位數|標準差', text) and numeric_collection:
        return pick('descriptive-statistics', 'explicit_statistic_with_values')
    if numeric and (has(r'\b(calculate|compute|evaluate|solve|simplify|integral|limit of|prime factors|similar .+ vectors|cosine similarity)\b|計算|求解', text)
                    or re.fullmatch(r'[\d\s+*/^().=−–\-i,%]+', text)):
        return pick('numeric-expression-evaluation', 'explicit_numerical_operation')
    if has(r'\b(translate|translation|transliterate|equivalent|say .+ in (?:french|spanish|german|english|chinese|japanese))\b|翻譯|音譯', text) and supplied:
        return pick('translation-alignment', 'translation_with_source_cue')
    if has(r'\b(rewrite|rephrase|paraphrase|correct|edit|proofread|shorten|expand|chang\w* .+ tense)\b|active to passive|passive to active|改寫|修正語法|精簡', text) and supplied:
        return pick('text-transformation', 'text_operation_with_source_cue')
    if has(r'\b(classify|categorize|label|sentiment|identify .+ (?:type|genre))\b|分類|情緒辨識', text) and supplied:
        return pick('classification-with-criteria', 'explicit_classification_and_item_cue')
    if has(r'\b(write|implement|debug|fix|design|generate|create|develop)\b|撰寫|實作|除錯',text) and has(r'\b(code|function|program|script|algorithm|sql|query|pseudocode|regex)\b|程式|演算法',text):
        return pick('code-task-framing','explicit_code_or_algorithm_task')
    if (has(r'\b(write|compose|draft|generate|create|invent|suggest|give|get|want)\b[^.!?]{0,100}\b(story|poem|poetry|limerick|haiku|dialogue|slogan|motto|headline|title|tagline|name|advertisement|ad copy|essay|letter|email|joke|insult)\b|創作.{0,20}(故事|詩|標語|標題)',text) or has(r'\b(good|suggest a|create a|write a)\s+(motto|slogan)\b',text)) and not has(r'\b(find|retrieve|fetch|email address|real quote)\b',text):
        return pick('constrained-creative-writing','explicit_creative_artifact')
    # External state is never inferred from a target's tool call or refusal.
    retrieval=has(r'\b(find|get|check|list|show|give|provide|obtain|retrieve|fetch|locate|discover|looking for|need|want|curious|tell me|are there|determine)\b',text)
    external=has(r'\b(whois|cpf|playlists?|weather|forecast|nearby|addresses?|standings|racecards|price|chart|news|events?|financials?|match|matches|players?|rankings?|ranks?|products?|jobs?|gifs?|images?|photos?|stats|statistics|route details|callback|currency|exchange rate|username|available|availability|profile|profiles|url|website|domains?|api|accounts?|assets?|flights?|restaurants?|hotels?|tickets?|stocks?|schedule|schedules|reservation|balance|users?|ids?|identifier|dividend|offer|rewards|ticker|data|details|zip|lat.long|coordinates|ip|leagues|team|fixtures|companies|company|funds|cryptocurrency|cryptocurrencies|earnings|revenue|feed|links|hashtag|soundtracks|market|markets|records|awards|view count|compatibility score|love match|horoscope|recipes|quote|cat fact|random fact|random skin|tax rate|charging|stations|hospital|schools|securities|demographic|songs?|films?|artists?|rate|water|code|taken|railway period|lottery)\b',text)
    if retrieval and external and not has(r'\b(how (?:do|does|can|to)|explain|definition|hypothetical|fictional|sample|example)\b',text):
        return pick('external-state-grounding','external_lookup_with_entity_or_state_cue')
    if retrieval and has(r'\b(apps|status|peer ratio|score|brands|places to eat|position|latest on|geographic location|post|backlinks|results|Euros .+ Dollars today)\b',text):
        return pick('external-state-grounding','external_result_or_inventory_request')
    if retrieval and has(r'\b(engines|inventory|catalogue|catalog)\b',text):
        return pick('external-state-grounding','external_inventory_lookup')
    if has(r'\b(local tax rate|coordinate .+ water)\b',text):
        return pick('external-state-grounding','location_specific_state_query')
    if has(r'\b(donate|upvote|create a poll|release the tool|examine expired certificates|extract .+ homepage)\b',text):
        return pick('external-state-grounding','explicit_external_state_operation')
    if has(r'\b(current|currently|latest|live|real.time|today|now|trending)\b|目前|即時|最新',text) and has(r'\b(price|stock|weather|score|statistics|rate|news|creator|trend|time|temperature|exchange)\b|股價|天氣|匯率|價格',text):
        return pick('external-state-grounding','explicit_time_sensitive_state')
    if has(r'(?:^|\b(?:please|can you|could you|i want to|i need to|try to)\s+)(book|reserve|order|purchase|buy|send|delete|download|upload|unsubscribe|cancel|register|retrieve|fetch|look up|search for|look for)\b|預訂|購買|發送|下載|查找',text) and not has(r'\b(how|guide|explain|steps|algorithm|code)\b|如何|教學',text):
        return pick('external-state-grounding','requested_external_lookup_or_action')
    if has(r'\b(find|get|check|list|show)\b',text) and has(r'\b(email|address|contact|profile|url|website|domain|api|account|asset|flight|restaurant|hotel|ticket|stock|schedule|reservation|password|balance|user|id)\b',text):
        return pick('external-state-grounding','external_entity_lookup')
    property_query = select_factual_property(model_input)
    if property_query is not None:
        return property_query
    if has(r'\b(?:propose|formulate|generate|develop|suggest|give|create)\b.{0,55}\b(?:hypothesis|hypotheses|explanation)\b|\bhypothesize\b|^predict\b|提出假說',text):
        return pick('hypothesis-and-evidence','explicit_hypothesis_or_prediction')
    if has(r'\b(compare|comparison|difference|different from|differs? from|versus|vs\.?|pros and cons|advantages and disadvantages|similarities)\b|比較|差異|優缺點',text):
        return pick('comparison-by-dimensions','explicit_comparison')
    if has(r'\b(how (?:can|should|could|do) (?:i|we|you)|how long should|how will you|ways? to|tips|advice|strategies|deal with|best way|recommendations for|create .+ business model)\b|\b(?:want to|need to|help me)\s+(?:improve|reduce|prevent|manage)\b|建議|如何改善|怎麼做',text):
        return pick('practical-guidance','explicit_goal_or_method')
    if has(r'\b(example|examples|ideas|recommend|suggest\w*|list|antonyms|synonyms|name (?:some|three|five)|what are some|give (?:me )?\w+ (?:examples|ways|reasons|benefits|activities))\b|列出|舉例|推薦|點子',text):
        return pick('option-generation','explicit_options_or_examples')
    if has(r'\b(explain|describe|concept|definition|define|history|origin|importance|significance|relationship|why|how (?:do|does|is|are|did|can))\b|解釋|原理|為什麼|歷史|什麼是|何謂',text):
        return pick('concept-explanation','explicit_explanation_or_mechanism')
    if has(r'(?:^|[.!]\s+|,\s*)(?:what|who|when|where|which|is|are|was|were|does|do|can|could|would|will|how many|how much|how long|how important)\b|\b(tell me (?:about|what|who)|give .+ definition)\b|^(?:誰|何時|哪裡|什麼|多少|是否)',text):
        return pick('factual-question-resolution','explicit_factual_question')
    if has(r'\b(need to know if|word .+ related|related to both)\b',text):
        return pick('factual-question-resolution','explicit_fact_or_lexical_relation')
    if re.fullmatch(r'I need to know about (?:a|an|the) \w+[.!]?',text,re.I):
        return pick('clarify-missing-inputs','underspecified_role_or_object_request')
    users=[m for m in messages if m['role']=='user']
    if len(users)>1 and has(r'\b(that|those|it|them|above|previous|continue|another|more|yes|sure|what about|how about)\b|繼續|剛才|上面|可以',text):
        return pick('conversation-reference-resolution','explicit_visible_conversation_reference')
    previous=messages[-2] if len(messages)>1 else None
    if previous and previous['role']=='assistant' and (has(r'\bplease provide\b|\bcould you (?:provide|specify|tell)\b|請提供',previous['content']) or '?' in previous['content']) and not has(r'\?|^(?:write|create|find|show|tell|please)\b',text):
        return pick('conversation-reference-resolution','reply_to_visible_clarification')
    return None
