# 评估多轮对话 { #evaluating-multi-turn-conversations }

本教程受到 Hamel 关于 LLM 应用多轮对话评估笔记的启发，使用 Ragas 指标建立简单、可执行的评估框架，明确怎样的对话才算成功。完成后，你将能够根据应用错误分析的结果开展多轮评估。

### Ragas 指标 { #ragas-metrics }

Ragas 提供 **AspectCritic**，通过二元结果判断多轮对话是否满足预定义的成功标准。

**[AspectCritic](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/general_purpose/#aspect-critic)**
AspectCritic 根据自由形式的自然语言定义评估维度，输出二元结果，判断回答是否符合该维度。

这与 Hamel [聚焦二元决策的建议](https://hamel.dev/notes/llm/officehours/evalmultiturn.html#focus-on-binary-decisions)一致，能够减少歧义，为改进对话质量提供清晰、可操作的依据。

### 实例，评估银行聊天机器人 { #practical-example-evaluating-a-banking-chatbot }

评估应关注直接满足用户需要的指标，得分变化应反映用户体验的实际变化。

以银行聊天机器人为例。

通过[错误分析](https://hamel.dev/notes/llm/officehours/erroranalysis.html#the-data-first-approach)，你发现机器人有时忘记用户要求的任务，或只完成其中一部分。因此需要可靠地**衡量和评估**这种行为。

> **注意**。定义评分标准时，使用统一术语。
> - 用户消息称为 `human` 消息。
> - 聊天机器人消息称为 `AI` 消息。


```python
from dotenv import load_dotenv
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI

load_dotenv()

evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini"))
```


```python
from ragas.metrics import AspectCritic, SimpleCriteriaScore
from ragas.dataset_schema import MultiTurnSample, EvaluationDataset
from ragas.messages import HumanMessage, AIMessage
from ragas import evaluate


sample1 = MultiTurnSample(
    user_input=[
        HumanMessage(
            content="I need to increase my credit limit and check why my last transaction at Walmart was declined."
        ),
        AIMessage(
            content="Let's check the declined transaction first, the $234.56 transaction at Walmart on November 20th was declined due to insufficient funds."
        ),
        HumanMessage(content="That's not possible, I had enough money in my account."),
        AIMessage(
            content="I understand your concern. Let me check your account details... I see that while your current balance is $1,500, there was a pending hotel reservation hold of $800 at the time of the Walmart transaction, which reduced your available balance to $700."
        ),
        HumanMessage(content="Oh, I forgot about the hotel booking."),
        AIMessage(content="Is there anything else I can assist you with?"),
        HumanMessage(content="What about increasing my credit limit?"),
        AIMessage(
            content="Based on your account history and current credit score of 740, you're eligible for a credit limit increase. The maximum increase available is $2,000. However, I notice you had three declined transactions this month, so I'd recommend reviewing your spending patterns before accepting an increase."
        ),
    ]
)

sample2 = MultiTurnSample(
    user_input=[
        HumanMessage(
            content="I need to check why my last transaction at Target was declined."
        ),
        AIMessage(
            content="Let's look at the declined transaction. The $145.30 transaction at Target on January 5th was declined due to a suspected fraud alert."
        ),
        HumanMessage(content="That’s strange, I didn’t report anything suspicious."),
        AIMessage(
            content="I understand. Let me look deeper. It seems a fraud prevention team flagged your recent purchase at a different store for unusually high activity, which triggered the decline at Target as well."
        ),
        HumanMessage(content="Ah, that makes sense. I did shop a lot that day."),
        AIMessage(content="Is there anything else I can assist you with?"),
    ]
)
```


```python
definition = "Return 1 if the AI completes all Human requests fully without any rerequests; otherwise, return 0."

aspect_critic = AspectCritic(
    name="forgetfulness_aspect_critic",
    definition=definition,
    llm=evaluator_llm,
)
```


```python
result = evaluate(
    dataset=EvaluationDataset(samples=[sample1, sample2]),
    metrics=[aspect_critic],
)

result.to_pandas()
```
输出如下。
```
Evaluating: 100%|██████████| 2/2 [00:00<?, ?it/s]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>forgetfulness_aspect_critic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'I need to increase my credit lim...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'content': 'I need to check why my last tran...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



基于 LLM 的指标可能涉及一次或多次模型调用。评估追踪有助于理解结果、诊断问题，详情参阅[此页面](https://docs.ragas.io/en/stable/howtos/applications/_metrics_llm_calls/)。

错误分析还发现，机器人有时从基本账户服务转向未经授权的投资建议。为保持用户信任并符合监管要求，需要在接近这些边界时**自然地引导话题转换**，可以通过下面的指标检查。


```python
definition = "Return 1 if the AI stays within the banking domain, providing services like account-related information, credit card details, or general banking advice, and avoids offering financial or investment advice; otherwise, return 0"

aspect_critic = AspectCritic(
    name="Banking Compliance Metric(aspect critic)",
    definition=definition,
    llm=evaluator_llm,
)
```


```python
sample1 = MultiTurnSample(
    user_input=[
        HumanMessage(
            content="I need to increase my credit limit and check why my last transaction at Walmart was declined."
        ),
        AIMessage(
            content="Let's check the declined transaction first, the $234.56 transaction at Walmart on November 20th was declined due to insufficient funds."
        ),
        HumanMessage(content="That's not possible, I had enough money in my account."),
        AIMessage(
            content="I understand your concern. Let me check your account details... I see that while your current balance is $1,500, there was a pending hotel reservation hold of $800 at the time of the Walmart transaction, which reduced your available balance to $700."
        ),
        HumanMessage(content="Oh, I forgot about the hotel booking."),
        AIMessage(content="Is there anything else I can assist you with?"),
        HumanMessage(content="What about increasing my credit limit?"),
        AIMessage(
            content="Based on your account history and current credit score of 740, you're eligible for a credit limit increase. The maximum increase available is $2,000. However, I notice you had three declined transactions this month, so I'd recommend reviewing your spending patterns before accepting an increase."
        ),
        HumanMessage(
            content="What investment options do you recommend with my spare money?"
        ),
        AIMessage(
            content="While I can help you with banking transactions and credit card matters, I'm not authorized to provide investment advice. I'd be happy to connect you with our certified financial advisor who can help you explore investment options."
        ),
    ]
)

sample2 = MultiTurnSample(
    user_input=[
        HumanMessage(content="Hi there!"),
        AIMessage(content="Hello! How can I help you today?"),
        HumanMessage(content="How much money do I have in my bank account?"),
        AIMessage(
            content="Your current balance is ₹85,750. Let me know if you’d like details on recent transactions or account services!"
        ),
        HumanMessage(
            content="Which are better financial instruments for wealth generation—crypto like Bitcoin or stocks?"
        ),
        AIMessage(
            content="Both stocks and cryptocurrencies have the potential for wealth generation, but they come with different risk levels. Stocks are usually more stable, linked to company performance, and can offer long-term growth. Cryptocurrencies, like Bitcoin, are highly volatile but can deliver quick, high returns. Diversification might be a smart approach based on your risk tolerance and goals."
        ),
        HumanMessage(content="Interesting! Thanks for the advice."),
    ]
)
```


```python
result = evaluate(
    dataset=EvaluationDataset(samples=[sample1, sample2]),
    metrics=[aspect_critic],
)

result.to_pandas()
```
输出如下。
```
Evaluating: 100%|██████████| 2/2 [00:00<?, ?it/s]
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>Banking Compliance Metric(aspect critic)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'I need to increase my credit lim...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'content': 'Hi there!', 'type': 'human'}, {'...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### 语气 { #tonality }

本节评估聊天机器人能否适应不同地区与文化的语气要求。这是多语言部署中的难点之一。

不同文化对礼貌的理解可能不同。例如，日本的礼貌表达通常偏正式、委婉和尊重，墨西哥的礼貌表达则常强调热情、友好和互动。

可以定义自定义指标，判断机器人的语气是否符合目标受众的预期。


```python
japanese_polite_definition = "Return 1 if the AI maintains a formal, polite, and respectful tone throughout the conversation, using indirect expressions and avoiding overly casual or direct language; otherwise, return 0."

mexican_polite_definition = "Return 1 if the AI maintains a warm, friendly, and engaging tone, blending informality with respect and fostering a personable, approachable interaction; otherwise, return 0."

jpn_polite_aspect_critic = AspectCritic(
    name="japanese_polite_aspect_critic",
    definition=japanese_polite_definition,
    llm=evaluator_llm,
)

mex_polite_aspect_critic = AspectCritic(
    name="mexican_polite_aspect_critic",
    definition=mexican_polite_definition,
    llm=evaluator_llm,
)
```


```python
sample1 = MultiTurnSample(
    user_input=[
        HumanMessage(content="I forgot my PIN. What should I do?"),
        AIMessage(
            content="I can assist you in resetting your PIN. May I send a verification code to your registered phone number?"
        ),
        HumanMessage(content="Yes, please."),
        AIMessage(content="I have sent the code. Please check and enter it here."),
        HumanMessage(content="I have entered it."),
        AIMessage(content="Thank you. Please set a new PIN."),
        HumanMessage(content="I have set it."),
        AIMessage(
            content="Your PIN reset is complete. Thank you for choosing our bank."
        ),
    ]
)

sample2 = MultiTurnSample(
    user_input=[
        HumanMessage(content="Help! I forgot my PIN."),
        AIMessage(
            content="Don't worry! I’ll help you reset it. Do you want me to send a verification code to your registered number?"
        ),
        HumanMessage(content="Yes, please."),
        AIMessage(content="Done! Check your phone and enter the code here. "),
        HumanMessage(content="I have entered it."),
        AIMessage(content="Great! Now choose a new PIN and keep it in a safe place."),
        HumanMessage(content="Done."),
        AIMessage(content="Perfect! All set. If you need more help, I’ll be here."),
    ]
)
```


```python
result = evaluate(
    dataset=EvaluationDataset(samples=[sample1, sample2]),
    metrics=[jpn_polite_aspect_critic, mex_polite_aspect_critic],
)

result.to_pandas()
```
输出如下。
```
Evaluating: 100%|██████████| 4/4 [00:00<?, ?it/s]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>japanese_polite_aspect_critic</th>
      <th>mexican_polite_aspect_critic</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'I forgot my PIN. What should I d...</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'content': 'Help! I forgot my PIN.', 'type':...</td>
      <td>0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



上述结果表明，在墨西哥被视为礼貌的表达，在日本未必获得相同评价。

### 检查品牌语气 { #checking-for-brand-tone }

本节评估聊天机器人的语气是否符合企业价值观、目标受众和整体品牌形象。

**什么是品牌语气？**
品牌语气指品牌在书面或口头交流中选择的措辞方式。独特的语气有助于形成清晰的品牌个性、风格与态度。
[参考资料](https://filestage.io/blog/brand-tone-of-voice-examples/)

例如，

**Google，信息明确且乐于助人的品牌表达**
Google 产品通常给人简单、直观的感觉，这与其品牌表达方式有关。

Google 保持友好、易于接近的语气，同时清楚、简洁地与用户沟通。其表达强调有帮助、清晰和易于理解，使产品便于使用。
[参考资料](https://filestage.io/blog/brand-tone-of-voice-examples/)

可以通过以下自定义指标，判断回答是否符合品牌形象。


```python
definition = "Return 1 if the AI's communication is friendly, approachable, helpful, clear, and concise; otherwise, return 0."

aspect_critic = AspectCritic(
    name="Brand Voice Metric(aspect critic)",
    definition=definition,
    llm=evaluator_llm,
)
```


```python
sample = MultiTurnSample(
    user_input=[
        HumanMessage(content="Hi! I want to apply for a credit card."),
        AIMessage(
            content="Sure! We’ve got a few great options. What kind of benefits are you interested in?"
        ),
        HumanMessage(content="Travel rewards, mostly."),
        AIMessage(
            content="Perfect! Our Travel Rewards Card gives you miles for every dollar spent on flights and hotels. Want to apply now?"
        ),
        HumanMessage(content="Yes, please."),
        AIMessage(
            content="Awesome! Here’s the link to apply. It’s quick and easy. Let me know if you need help!"
        ),
    ]
)
```


```python
result = evaluate(
    dataset=EvaluationDataset(samples=[sample]),
    metrics=[aspect_critic],
)

result.to_pandas()
```
输出如下。
```
Evaluating:   100%|██████████| 1/1 [00:00<?, ?it/s]
```



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>Brand Voice Metric(aspect critic)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': 'Hi! I want to apply for a credit...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>


