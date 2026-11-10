
# 创建并评估集成知识库与操作组的 Amazon Bedrock 智能体 { #create-and-evaluate-an-amazon-bedrock-agent-integrated-with-an-amazon-bedrock-knowledge-base-and-action-groups }

本教程评估一个 Amazon Bedrock 餐厅智能体。它提供成人与儿童菜单信息，并管理订座。示例基于 [Amazon Bedrock Agents](https://aws.amazon.com/bedrock/agents/) 的[功能示例](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/agents-and-function-calling/bedrock-agents/features-examples/05-create-agent-with-knowledge-base-and-action-group)稍作修改，创建流程也可参阅该示例。

架构如下。

![系统架构](../../../_static/architecture.png)

本教程包括以下步骤。

- 导入所需库
- 创建智能体
- 定义 Ragas 指标
- 评估智能体
- 清理创建的资源

??? note "点击查看智能体创建过程"
    ## 导入所需库 { #import-the-needed-libraries }

    首先安装依赖。


    ```python
    %pip install --upgrade -q boto3 opensearch-py botocore awscli retrying ragas langchain-aws
    ```

    以下命令克隆包含本教程辅助文件的仓库。

    ```
    ! git clone https://huggingface.co/datasets/vibrantlabsai/booking_agent_utils
    ```


    ```python
    import os
    import time
    import boto3
    import logging
    import pprint
    import json

    from booking_agent_utils.knowledge_base import BedrockKnowledgeBase
    from booking_agent_utils.agent import (
        create_agent_role_and_policies,
        create_lambda_role,
        delete_agent_roles_and_policies,
        create_dynamodb,
        create_lambda,
        clean_up_resources,
    )
    ```


    ```python
    # Clients
    s3_client = boto3.client("s3")
    sts_client = boto3.client("sts")
    session = boto3.session.Session()
    region = session.region_name
    account_id = sts_client.get_caller_identity()["Account"]
    bedrock_agent_client = boto3.client("bedrock-agent")
    bedrock_agent_runtime_client = boto3.client("bedrock-agent-runtime")
    logging.basicConfig(
        format="[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    region, account_id
    ```


    ```python
    suffix = f"{region}-{account_id}"
    agent_name = "booking-agent"
    knowledge_base_name = f"{agent_name}-kb"
    knowledge_base_description = (
        "Knowledge Base containing the restaurant menu's collection"
    )
    agent_alias_name = "booking-agent-alias"
    bucket_name = f"{agent_name}-{suffix}"
    agent_bedrock_allow_policy_name = f"{agent_name}-ba"
    agent_role_name = f"AmazonBedrockExecutionRoleForAgents_{agent_name}"
    agent_foundation_model = "amazon.nova-pro-v1:0"

    agent_description = "Agent in charge of a restaurants table bookings"
    agent_instruction = """
    You are a restaurant agent responsible for managing clients’ bookings (retrieving, creating, or canceling reservations) and assisting with menu inquiries. When handling menu requests, provide detailed information about the requested items. Offer recommendations only when:

    1. The customer explicitly asks for a recommendation, even if the item is available (include complementary dishes).
    2. The requested item is unavailable—inform the customer and suggest suitable alternatives.
    3. For general menu inquiries, provide the full menu and add a recommendation only if the customer asks for one.

    In all cases, ensure that any recommended items are present in the menu.

    Ensure all responses are clear, contextually relevant, and enhance the customer's experience.
    """

    agent_action_group_description = """
    Actions for getting table booking information, create a new booking or delete an existing booking"""

    agent_action_group_name = "TableBookingsActionGroup"
    ```

    ## 配置智能体 { #setting-up-agent }

    ### 创建 Amazon Bedrock 知识库 { #create-knowledge-base-for-amazon-bedrock }

    先创建 [Amazon Bedrock 知识库](https://aws.amazon.com/bedrock/knowledge-bases/)保存餐厅菜单，本例与 Amazon OpenSearch Serverless 集成。


    ```python
    knowledge_base = BedrockKnowledgeBase(
        kb_name=knowledge_base_name,
        kb_description=knowledge_base_description,
        data_bucket_name=bucket_name,
    )
    ```

    ### 将数据集上传到 Amazon S3 { #upload-the-dataset-to-amazon-s3 }
    接下来通过辅助类调用 [boto3 API](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/start_ingestion_job.html)，将菜单数据加入知识库。

    先把 dataset 文件夹中的菜单上传到 Amazon S3。


    ```python
    def upload_directory(path, bucket_name):
        for root, dirs, files in os.walk(path):
            for file in files:
                file_to_upload = os.path.join(root, file)
                print(f"uploading file {file_to_upload} to {bucket_name}")
                s3_client.upload_file(file_to_upload, bucket_name, file)


    upload_directory("booking_agent_utils/dataset", bucket_name)
    ```

    然后启动数据摄取任务。


    ```python
    # ensure that the kb is available
    time.sleep(30)
    # sync knowledge base
    knowledge_base.start_ingestion_job()
    ```

    最后记录知识库 ID，后续将其关联到智能体。


    ```python
    kb_id = knowledge_base.get_knowledge_base_id()
    ```

    #### 使用 Retrieve and Generate API 测试知识库 { #testing-knowledge-base-with-retrieve-and-generate-api }

    调用 Retrieve and Generate API，检查知识库是否正常工作。


    ```python
    response = bedrock_agent_runtime_client.retrieve_and_generate(
        input={"text": "Which are the mains available in the childrens menu?"},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": kb_id,
                "modelArn": "arn:aws:bedrock:{}::foundation-model/{}".format(
                    region, agent_foundation_model
                ),
                "retrievalConfiguration": {
                    "vectorSearchConfiguration": {"numberOfResults": 5}
                },
            },
        },
    )

    print(response["output"]["text"], end="\n" * 2)
    ```

    ### 创建 DynamoDB 表 { #create-the-dynamodb-table }

    创建保存餐厅预订信息的 DynamoDB 表。


    ```python
    table_name = "restaurant_bookings"
    create_dynamodb(table_name)
    ```

    ### 创建 Lambda 函数 { #create-the-lambda-function }

    创建与 DynamoDB 表交互的 Lambda 函数。

    #### 编写函数代码 { #create-the-function-code }

    实现 `get_booking_details`、`create_booking` 和 `delete_booking`。


    ```python
    %%writefile lambda_function.py
    import json
    import uuid
    import boto3

    dynamodb = boto3.resource('dynamodb')
    table = dynamodb.Table('restaurant_bookings')

    def get_named_parameter(event, name):
        """
        Get a parameter from the lambda event
        """
        return next(item for item in event['parameters'] if item['name'] == name)['value']


    def get_booking_details(booking_id):
        """
        Retrieve details of a restaurant booking

        Args:
            booking_id (string): The ID of the booking to retrieve
        """
        try:
            response = table.get_item(Key={'booking_id': booking_id})
            if 'Item' in response:
                return response['Item']
            else:
                return {'message': f'No booking found with ID {booking_id}'}
        except Exception as e:
            return {'error': str(e)}


    def create_booking(date, name, hour, num_guests):
        """
        Create a new restaurant booking

        Args:
            date (string): The date of the booking
            name (string): Name to idenfity your reservation
            hour (string): The hour of the booking
            num_guests (integer): The number of guests for the booking
        """
        try:
            booking_id = str(uuid.uuid4())[:8]
            table.put_item(
                Item={
                    'booking_id': booking_id,
                    'date': date,
                    'name': name,
                    'hour': hour,
                    'num_guests': num_guests
                }
            )
            return {'booking_id': booking_id}
        except Exception as e:
            return {'error': str(e)}


    def delete_booking(booking_id):
        """
        Delete an existing restaurant booking

        Args:
            booking_id (str): The ID of the booking to delete
        """
        try:
            response = table.delete_item(Key={'booking_id': booking_id})
            if response['ResponseMetadata']['HTTPStatusCode'] == 200:
                return {'message': f'Booking with ID {booking_id} deleted successfully'}
            else:
                return {'message': f'Failed to delete booking with ID {booking_id}'}
        except Exception as e:
            return {'error': str(e)}


    def lambda_handler(event, context):
        # get the action group used during the invocation of the lambda function
        actionGroup = event.get('actionGroup', '')

        # name of the function that should be invoked
        function = event.get('function', '')

        # parameters to invoke function with
        parameters = event.get('parameters', [])

        if function == 'get_booking_details':
            booking_id = get_named_parameter(event, "booking_id")
            if booking_id:
                response = str(get_booking_details(booking_id))
                responseBody = {'TEXT': {'body': json.dumps(response)}}
            else:
                responseBody = {'TEXT': {'body': 'Missing booking_id parameter'}}

        elif function == 'create_booking':
            date = get_named_parameter(event, "date")
            name = get_named_parameter(event, "name")
            hour = get_named_parameter(event, "hour")
            num_guests = get_named_parameter(event, "num_guests")

            if date and hour and num_guests:
                response = str(create_booking(date, name, hour, num_guests))
                responseBody = {'TEXT': {'body': json.dumps(response)}}
            else:
                responseBody = {'TEXT': {'body': 'Missing required parameters'}}

        elif function == 'delete_booking':
            booking_id = get_named_parameter(event, "booking_id")
            if booking_id:
                response = str(delete_booking(booking_id))
                responseBody = {'TEXT': {'body': json.dumps(response)}}
            else:
                responseBody = {'TEXT': {'body': 'Missing booking_id parameter'}}

        else:
            responseBody = {'TEXT': {'body': 'Invalid function'}}

        action_response = {
            'actionGroup': actionGroup,
            'function': function,
            'functionResponse': {
                'responseBody': responseBody
            }
        }

        function_response = {'response': action_response, 'messageVersion': event['messageVersion']}
        print("Response: {}".format(function_response))

        return function_response
    ```

    #### 创建所需权限 { #create-the-required-permissions }


    ```python
    lambda_iam_role = create_lambda_role(agent_name, table_name)
    ```

    #### 创建函数 { #create-the-function }


    ```python
    lambda_function_name = f"{agent_name}-lambda"
    lambda_function = create_lambda(lambda_function_name, lambda_iam_role)
    ```

    ### 创建智能体所需的 IAM 策略 { #create-the-iam-policies-needed-for-the-agent }

    知识库、DynamoDB 表和执行任务的 Lambda 函数已准备好，下面创建智能体。


    ```python
    agent_role = create_agent_role_and_policies(
        agent_name, agent_foundation_model, kb_id=kb_id
    )
    ```

    ### 创建智能体 { #create-the-agent }

    IAM 角色创建后，通过 boto3 的 [`create_agent`](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/create_agent.html) API 创建智能体。


    ```python
    response = bedrock_agent_client.create_agent(
        agentName=agent_name,
        agentResourceRoleArn=agent_role["Role"]["Arn"],
        description=agent_description,
        idleSessionTTLInSeconds=1800,
        foundationModel=agent_foundation_model,
        instruction=agent_instruction,
    )
    ```

    获取智能体 ID，用于后续操作。


    ```python
    agent_id = response["agent"]["agentId"]
    print("The agent id is:", agent_id)
    ```

    ### 创建智能体操作组 { #create-the-agent-action-group }

    创建使用前述 Lambda 函数的操作组，通过描述向智能体说明它能执行的功能。

    函数 schema 需要包含每个函数的名称、描述和参数。


    ```python
    agent_functions = [
        {
            "name": "get_booking_details",
            "description": "Retrieve details of a restaurant booking",
            "parameters": {
                "booking_id": {
                    "description": "The ID of the booking to retrieve",
                    "required": True,
                    "type": "string",
                }
            },
        },
        {
            "name": "create_booking",
            "description": "Create a new restaurant booking",
            "parameters": {
                "date": {
                    "description": "The date of the booking",
                    "required": True,
                    "type": "string",
                },
                "name": {
                    "description": "Name to idenfity your reservation",
                    "required": True,
                    "type": "string",
                },
                "hour": {
                    "description": "The hour of the booking",
                    "required": True,
                    "type": "string",
                },
                "num_guests": {
                    "description": "The number of guests for the booking",
                    "required": True,
                    "type": "integer",
                },
            },
        },
        {
            "name": "delete_booking",
            "description": "Delete an existing restaurant booking",
            "parameters": {
                "booking_id": {
                    "description": "The ID of the booking to delete",
                    "required": True,
                    "type": "string",
                }
            },
        },
    ]
    ```

    使用 schema 调用 [`create_agent_action_group`](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agent/client/create_agent_action_group.html)，创建操作组。


    ```python
    # Pause to make sure agent is created
    time.sleep(30)

    # Now, we can configure and create an action group here:
    agent_action_group_response = bedrock_agent_client.create_agent_action_group(
        agentId=agent_id,
        agentVersion="DRAFT",
        actionGroupExecutor={"lambda": lambda_function["FunctionArn"]},
        actionGroupName=agent_action_group_name,
        functionSchema={"functions": agent_functions},
        description=agent_action_group_description,
    )
    ```

    ### 允许智能体调用操作组 Lambda { #allow-the-agent-to-invoke-the-action-group-lambda }


    ```python
    # Create allow to invoke permission on lambda
    lambda_client = boto3.client("lambda")
    response = lambda_client.add_permission(
        FunctionName=lambda_function_name,
        StatementId="allow_bedrock",
        Action="lambda:InvokeFunction",
        Principal="bedrock.amazonaws.com",
        SourceArn=f"arn:aws:bedrock:{region}:{account_id}:agent/{agent_id}",
    )
    ```

    ### 将知识库关联到智能体 { #associate-the-knowledge-base-to-the-agent }


    ```python
    response = bedrock_agent_client.associate_agent_knowledge_base(
        agentId=agent_id,
        agentVersion="DRAFT",
        description="Access the knowledge base when customers ask about the plates in the menu.",
        knowledgeBaseId=kb_id,
        knowledgeBaseState="ENABLED",
    )
    ```

    ### 准备智能体并创建别名 { #prepare-the-agent-and-create-an-alias }

    创建可用于内部测试的 DRAFT 版本。


    ```python
    response = bedrock_agent_client.prepare_agent(agentId=agent_id)
    print(response)
    # Pause to make sure agent is prepared
    time.sleep(30)
    ```


    ```python
    response = bedrock_agent_client.create_agent_alias(
        agentAliasName="TestAlias",
        agentId=agent_id,
        description="Test alias",
    )

    alias_id = response["agentAlias"]["agentAliasId"]
    print("The Agent alias is:", alias_id)
    time.sleep(30)
    ```

`invokeAgent` 将用户查询发送给 Bedrock 智能体，并处理事件流，返回回答及用于评估的追踪数据。

```python
def invokeAgent(query, session_id, session_state=dict()):
    end_session: bool = False

    # invoke the agent API
    agentResponse = bedrock_agent_runtime_client.invoke_agent(
        inputText=query,
        agentId=agent_id,
        agentAliasId=alias_id,
        sessionId=session_id,
        enableTrace=True,
        endSession=end_session,
        sessionState=session_state,
    )

    event_stream = agentResponse["completion"]
    try:
        traces = []
        for event in event_stream:
            if "chunk" in event:
                data = event["chunk"]["bytes"]
                agent_answer = data.decode("utf8")
                end_event_received = True
                return agent_answer, traces
                # End event indicates that the request finished successfully
            elif "trace" in event:
                traces.append(event["trace"])
            else:
                raise Exception("unexpected event.", event)
        return agent_answer, traces
    except Exception as e:
        raise Exception("unexpected event.", e)
```

## 定义 Ragas 指标 { #defining-the-ragas-metrics }

智能体会执行复杂任务，且往往存在多种合理路径，因此评估需要关注多个维度。

智能体具有自主性，更需要评估来确认其行为符合预期。

#### 选择评估内容 { #choosing-what-to-evaluate-in-your-agent }

指标取决于场景，优先选择直接对应用户需求或业务价值的维度。本例希望餐厅智能体完整处理请求、避免多余重复、适时提供推荐，并保持品牌语气。

下面为这些目标定义指标。Ragas 支持多种用户自定义评估方式。

评判标准应采用定义明确的二元决策或离散类别，促使你清楚界定成功条件。缺少解释依据的 0～100 分量表往往难以区分相近结果，例如 87 与 91 分，尤其在独立评估时。

下面使用两类适合的指标。

- [Aspect Critic](../../concepts/metrics/available_metrics/aspect_critic.md)，通过 LLM 判断是否满足用户标准，返回二元结果。
- [评分细则指标](../../concepts/metrics/available_metrics/general_purpose.md#rubrics-based-criteria-scoring)，根据用户定义的详细标准一致地评价质量。


```python
from langchain_aws import ChatBedrock
from ragas.llms import LangchainLLMWrapper

model_id = "us.amazon.nova-pro-v1:0"   # Choose your desired model
region_name = "us-east-1"              # Choose your desired AWS region

bedrock_llm = ChatBedrock(model_id=model_id, region_name=region_name)
evaluator_llm = LangchainLLMWrapper(bedrock_llm)
```


```python
from ragas.metrics import AspectCritic, RubricsScore
from ragas.dataset_schema import SingleTurnSample, MultiTurnSample, EvaluationDataset
from ragas import evaluate

rubrics = {
    "score-1_description": (
        "The item requested by the customer is not present in the menu and no recommendations were made."
    ),
    "score0_description": (
        "Either the item requested by the customer is present in the menu, or the conversation does not include any food or menu inquiry (e.g., booking, cancellation). This score applies regardless of whether any recommendation was provided."
    ),
    "score1_description": (
        "The item requested by the customer is not present in the menu and a recommendation was provided."
    ),
}

recommendations = RubricsScore(rubrics=rubrics, llm=evaluator_llm, name="Recommendations")


# Metric to evaluate if the AI fulfills all human requests completely.
request_completeness = AspectCritic(
    name="Request Completeness",
    llm=evaluator_llm,
    definition=(
        "Return 1 The agent completely fulfills all the user requests with no omissions. "
        "otherwise, return 0."
    ),
)

# Metric to assess if the AI's communication aligns with the desired brand voice.
brand_tone = AspectCritic(
    name="Brand Voice Metric",
    llm=evaluator_llm,
    definition=(
        "Return 1 if the AI's communication is friendly, approachable, helpful, clear, and concise; "
        "otherwise, return 0."
    ),
)
```

## 使用 Ragas 评估智能体 { #evaluating-agent-with-ragas }

先将追踪转换为 Ragas 格式。使用 [convert_to_ragas_messages][ragas.integrations.amazon_bedrock.convert_to_ragas_messages] 将 Bedrock 消息转为评估所需结构。详情参阅[评估数据集](../../concepts/components/eval_dataset.md)。


```python
%%time
import uuid
session_id:str = str(uuid.uuid1())
query = "If you have children food then book a table for 2 people at 7pm on the 5th of May 2025."
agent_answer, traces_1 = invokeAgent(query, session_id)

print(agent_answer)
```
输出如下。
```
Your booking for 2 people at 7pm on the 5th of May 2025 has been successfully created. Your booking ID is ca2fab70.
```

```python
query = "Can you check my previous booking? Can you please delete the booking?"
agent_answer, traces_2 = invokeAgent(query, session_id)

print(agent_answer)
```
输出如下。
```
Your reservation was found and has been successfully canceled.
```

```python
from ragas.integrations.amazon_bedrock import convert_to_ragas_messages

# Convert Amazon Bedrock traces to messages accepted by Ragas.
# The convert_to_ragas_messages function transforms Bedrock-specific trace data
# into a format that Ragas can process as conversation messages.
ragas_messages_trace_1 = convert_to_ragas_messages(traces_1)
ragas_messages_trace_2 = convert_to_ragas_messages(traces_2)

# Initialize MultiTurnSample objects.
# MultiTurnSample is a data type defined in Ragas that encapsulates conversation
# data for multi-turn evaluation. This conversion is necessary to perform evaluations.
sample_1 = MultiTurnSample(user_input=ragas_messages_trace_1)
sample_2 = MultiTurnSample(user_input=ragas_messages_trace_2)

result = evaluate(
    # Create an evaluation dataset from the multi-turn samples
    dataset=EvaluationDataset(samples=[sample_1, sample_2]),
    metrics=[request_completeness, brand_tone],
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
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>Request Completeness</th>
      <th>Brand Voice Metric</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': '[{text=If you have children food...</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'content': '[{text=If you have children food...</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



两段对话在完整性与品牌语气上都得 1 分，因为智能体完整处理全部请求，并以友好、易于理解、清楚简洁的方式沟通。


```python
%%time
import uuid

session_id:str = str(uuid.uuid1())
query = "Do you serve Chicken Wings?"

agent_answer, traces_3 = invokeAgent(query, session_id)
print(agent_answer)
```
输出如下。
```
Yes, we serve Chicken Wings. Here are the details:
- **Buffalo Chicken Wings**: Classic buffalo wings served with celery sticks and blue cheese dressing. Allergens: Dairy (in blue cheese dressing), Gluten (in the coating), possible Soy (in the sauce).
```

```python
%%time
session_id:str = str(uuid.uuid1())
query = "For desserts, do you have chocolate truffle cake?"
agent_answer, traces_4 = invokeAgent(query, session_id)
print(agent_answer)
```
输出如下。
```
I'm sorry, but we do not have chocolate truffle cake on our dessert menu. However, we have several delicious alternatives you might enjoy:

1. **Classic New York Cheesecake** - Creamy cheesecake with a graham cracker crust, topped with a choice of fruit compote or chocolate ganache.
2. **Apple Pie à la Mode** - Warm apple pie with a flaky crust, served with a scoop of vanilla ice cream and a drizzle of caramel sauce.
3. **Chocolate Lava Cake** - Rich and gooey chocolate cake with a molten center, dusted with powdered sugar and served with a scoop of raspberry sorbet.
4. **Pecan Pie Bars** - Buttery shortbread crust topped with a gooey pecan filling, cut into bars for easy serving.
5. **Banana Pudding Parfait** - Layers of vanilla pudding, sliced bananas, and vanilla wafers, topped with whipped cream and a sprinkle of crushed nuts.

May I recommend the **Chocolate Lava Cake** for a decadent treat?
```

```python
%%time
from datetime import datetime
today = datetime.today().strftime('%b-%d-%Y')

session_id:str = str(uuid.uuid1())
query = "Do you have indian food?"
session_state = {
    "promptSessionAttributes": {
        "name": "John",
        "today": today
    }
}

agent_answer, traces_5 = invokeAgent(query, session_id, session_state=session_state)
print(agent_answer)
```
输出如下。
```
I could not find Indian food on our menu. However, we offer a variety of other cuisines including American, Italian, and vegetarian options. Would you like to know more about these options?
```

```python
from ragas.integrations.amazon_bedrock import convert_to_ragas_messages

ragas_messages_trace_3 = convert_to_ragas_messages(traces_3)
ragas_messages_trace_4 = convert_to_ragas_messages(traces_4)
ragas_messages_trace_5 = convert_to_ragas_messages(traces_5)

sample_3 = MultiTurnSample(user_input=ragas_messages_trace_3)
sample_4 = MultiTurnSample(user_input=ragas_messages_trace_4)
sample_5 = MultiTurnSample(user_input=ragas_messages_trace_5)

result = evaluate(
    dataset=EvaluationDataset(samples=[sample_3, sample_4, sample_5]),
    metrics=[recommendations],
)

result.to_pandas()
```
```
Evaluating: 100%|██████████| 3/3 [00:00<?, ?it/s]
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
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>Recommendations</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': '[{text=Do you serve Chicken Wing...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[{'content': '[{text=For desserts, do you have...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[{'content': '[{text=Do you have indian food?}...</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



推荐指标中，鸡翅问题得 0 分，因为该商品有售；巧克力松露蛋糕和印度菜问题得 1 分，因为菜单没有相应商品，智能体提供了替代建议。

使用 Ragas 的 RAG 指标，还可以评估智能体对知识库检索信息的使用情况。更多背景见[指标文档]()。

本例采用以下 RAG 指标。

- [ContextRelevance](../../concepts/metrics/available_metrics/nvidia_metrics.md#context-relevance)，通过两次 LLM 评判，衡量检索上下文与用户查询的相关程度。
- [忠实度](../../concepts/metrics/available_metrics/faithfulness.md)，判断回答中的全部陈述是否得到检索上下文支持。
- [回答依据充分性](../../concepts/metrics/available_metrics/nvidia_metrics.md#response-groundedness)，衡量回答陈述在给定上下文中的依据。


```python
from ragas.metrics import ContextRelevance, Faithfulness,  ResponseGroundedness

metrics = [
    ContextRelevance(llm=evaluator_llm),
    Faithfulness(llm=evaluator_llm),
    ResponseGroundedness(llm=evaluator_llm),
]
```


```python
from ragas.integrations.amazon_bedrock import extract_kb_trace

kb_trace_3 = extract_kb_trace(traces_3)
kb_trace_4 = extract_kb_trace(traces_4)

trace_3_single_turn_sample = SingleTurnSample(
    user_input=kb_trace_3[0].get("user_input"),
    retrieved_contexts=kb_trace_3[0].get("retrieved_contexts"),
    response=kb_trace_3[0].get("response"),
    reference="Yes, we do serve chicken wings prepared in Buffalo style, chicken wing that’s typically deep-fried and then tossed in a tangy, spicy Buffalo sauce.",
)

trace_4_single_turn_sample = SingleTurnSample(
    user_input=kb_trace_4[0].get("user_input"),
    retrieved_contexts=kb_trace_4[0].get("retrieved_contexts"),
    response=kb_trace_4[0].get("response"),
    reference="The desserts on the adult menu are:\n1. Classic New York Cheesecake\n2. Apple Pie à la Mode\n3. Chocolate Lava Cake\n4. Pecan Pie Bars\n5. Banana Pudding Parfait",
)

single_turn_samples = [trace_3_single_turn_sample, trace_4_single_turn_sample]

dataset = EvaluationDataset(samples=single_turn_samples)
```


```python
kb_results = evaluate(dataset=dataset, metrics=metrics)
kb_results.to_pandas()
```
```
Evaluating: 100%|██████████| 6/6 [00:00<?, ?it/s]
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
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>retrieved_contexts</th>
      <th>response</th>
      <th>reference</th>
      <th>nv_context_relevance</th>
      <th>faithfulness</th>
      <th>nv_response_groundedness</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Chicken Wings</td>
      <td>[The Regrettable Experience -- Dinner Menu Ent...</td>
      <td>Yes, we serve Chicken Wings. Here are the deta...</td>
      <td>Yes, we do serve chicken wings prepared in Buf...</td>
      <td>1.0</td>
      <td>1.00</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>chocolate truffle cake</td>
      <td>[Allergens: Gluten (in the breading).     3. B...</td>
      <td>I'm sorry, but we do not have chocolate truffl...</td>
      <td>The desserts on the adult menu are:\n1. Classi...</td>
      <td>0.0</td>
      <td>0.75</td>
      <td>0.5</td>
    </tr>
  </tbody>
</table>
</div>

判断用户目标是否达成，可以使用以下指标。

- [AgentGoalAccuracyWithReference](../../concepts/metrics/available_metrics/agents.md#agent-goal-accuracy)，将最终结果与标注的理想结果比较，返回二元结果。
- [AgentGoalAccuracyWithoutReference](../../concepts/metrics/available_metrics/agents.md#agent-goal-accuracy)，根据对话推断用户目标与实际结果，判断是否成功，无须显式参考答案。


```python
from ragas.metrics import (
    AgentGoalAccuracyWithoutReference,
    AgentGoalAccuracyWithReference,
)

goal_accuracy_with_reference = AgentGoalAccuracyWithReference(llm=evaluator_llm)
goal_accuracy_without_reference = AgentGoalAccuracyWithoutReference(llm=evaluator_llm)
```


```python
%%time
import uuid

session_id:str = str(uuid.uuid1())
query = "What entrees do you have for children?"

agent_answer, traces_6 = invokeAgent(query, session_id)
print(agent_answer)
```
输出如下。
```
Here are the entrees available for children:
1. CHICKEN NUGGETS - Crispy chicken nuggets served with a side of ketchup or ranch dressing. Allergens: Gluten (in the coating), possible Soy. Suitable for Vegetarians: No
2. MACARONI AND CHEESE - Classic macaroni pasta smothered in creamy cheese sauce. Allergens: Dairy, Gluten. Suitable for Vegetarians: Yes
3. MINI CHEESE QUESADILLAS - Small flour tortillas filled with melted cheese, served with a mild salsa. Allergens: Dairy, Gluten. Suitable for Vegetarians: Yes
4. PEANUT BUTTER AND BANANA SANDWICH - Peanut butter and banana slices on whole wheat bread. Allergens: Nuts (peanut), Gluten. Suitable for Vegetarians: Yes (if using vegetarian peanut butter)
5. VEGGIE PITA POCKETS - Mini whole wheat pita pockets filled with hummus, cucumber, and cherry tomatoes. Allergens: Gluten, possible Soy. Suitable for Vegetarians: Yes
```

```python
from ragas.integrations.amazon_bedrock import convert_to_ragas_messages

ragas_messages_trace_6 = convert_to_ragas_messages(traces_6)

sample_6 = MultiTurnSample(
    user_input=ragas_messages_trace_6,
    reference="Response contains entrees food items for the children.",
)

result = evaluate(
    dataset=EvaluationDataset(samples=[sample_6]),
    metrics=[goal_accuracy_with_reference],
)

result.to_pandas()
```
```
Evaluating: 100%|██████████| 1/1 [00:00<?, ?it/s]
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
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>reference</th>
      <th>agent_goal_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': '[{text=What entrees do you have ...</td>
      <td>The final outcome provides child-friendly entr...</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
sample_6 = MultiTurnSample(user_input=ragas_messages_trace_6)

result = evaluate(
    dataset=EvaluationDataset(samples=[sample_6]),
    metrics=[goal_accuracy_without_reference],
)

result.to_pandas()
```
```
Evaluating: 100%|██████████| 1/1 [00:00<?, ?it/s]
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
<table border="1">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>user_input</th>
      <th>agent_goal_accuracy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[{'content': '[{text=What entrees do you have ...</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
</div>



两个场景都完整列出了儿童主菜选项，因此得分均为 1。

## 清理资源 { #clean-up }
删除本教程创建的关联资源，避免继续产生费用。

```python
clean_up_resources(
    table_name,
    lambda_function,
    lambda_function_name,
    agent_action_group_response,
    agent_functions,
    agent_id,
    kb_id,
    alias_id,
)
```

```python
# Delete the agent roles and policies
delete_agent_roles_and_policies(agent_name)
```

```python
# delete KB
knowledge_base.delete_kb(delete_s3_bucket=True, delete_iam_roles_and_policies=True)
```
