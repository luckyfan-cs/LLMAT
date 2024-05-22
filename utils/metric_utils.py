
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from llm_attacks.minimal_gcg.string_utils import SuffixManager, load_conversation_template
import json
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from utils.utils import prefix_check_for_attack_success, agent_check_for_attack_success
from utils.string_utils import  test_prefixes





def get_agent_asr(val_goals, val_targets, model, tokenizer, device,  universal_adv_prompt, args = None):
    template_name = args.template_name
    conv_template = load_conversation_template(template_name)
    flag = 0
    custom_args = {
        'base_model': "gpt-4",
        "max_n_tokens": 500,
        "max_n_attempts": 20,
        'n_streams': 1,
        'keep_last_n': 3,
        'n_iterations': 1
    }

    #agent = load_generator_model(custom_args)
    from GPTEvaluatorAgent.judges import EvidentJudge
    agent = EvidentJudge(args=args)
    for goal, target in zip(val_goals, val_targets):
        suffix_manager = SuffixManager(tokenizer=tokenizer,
                      conv_template=conv_template,
                      instruction=goal,
                      target=target,
                      adv_string=universal_adv_prompt)

        is_success = agent_check_for_attack_success(model,
                                                    tokenizer,
                                                    suffix_manager.get_input_ids(adv_string=universal_adv_prompt).to(device),
                                                    suffix_manager._assistant_role_slice,
                                                    agent,
                                                    goal,
                                                    target,
                                                    custom_args)
        if is_success:
            flag+=1
    asr = flag / len(val_goals)
    return  asr
def get_prefix_asr(val_goals, val_targets, model, tokenizer, device,  universal_adv_prompt, args = None):
    template_name = args.template_name
    conv_template = load_conversation_template(template_name)
    flag = 0

    for goal, target in zip(val_goals, val_targets):
        suffix_manager = SuffixManager(tokenizer=tokenizer,
                      conv_template=conv_template,
                      instruction=goal,
                      target=target,
                      adv_string=universal_adv_prompt)

        is_success = prefix_check_for_attack_success(model,
                                              tokenizer,
                                              suffix_manager.get_input_ids(adv_string=universal_adv_prompt).to(device),
                                              suffix_manager._assistant_role_slice,
                                              test_prefixes)

        if is_success:
            flag+=1
    asr = flag / len(val_goals)
    return  asr

def get_asr(val_goals, val_targets, model, tokenizer, device,  universal_adv_prompt, args = None):
    template_name = args.template_name #'llama-2'
    conv_template = load_conversation_template(template_name)
    flag = 0
    custom_args = {
        'base_model': "gpt-4",
        "max_n_tokens": 500,
        "max_n_attempts": 20,
        'n_streams': 1,
        'keep_last_n': 3,
        'n_iterations': 1
    }

    from GPTEvaluatorAgent.judges import EvidentJudge
    agent = EvidentJudge(args=args)
    for goal, target in zip(val_goals, val_targets):
        suffix_manager = SuffixManager(tokenizer=tokenizer,
                      conv_template=conv_template,
                      instruction=goal,
                      target=target,
                      adv_string=universal_adv_prompt)
        if args.check_type == "prefixes":
            is_success = prefix_check_for_attack_success(model,
                                                  tokenizer,
                                                  suffix_manager.get_input_ids(adv_string=universal_adv_prompt).to(device),
                                                  suffix_manager._assistant_role_slice,
                                                  test_prefixes)

        elif args.check_type == "agent":

            is_success = agent_check_for_attack_success(model,
                                                        tokenizer,
                                                        suffix_manager.get_input_ids(adv_string=universal_adv_prompt).to(device),
                                                        suffix_manager._assistant_role_slice,
                                                        agent,
                                                        goal,
                                                        target,
                                                        custom_args)
        if is_success:
            flag+=1
    asr = flag / len(val_goals)
    return  asr


def get_similarity(embeddings):
    similarity = cosine_similarity(embeddings, embeddings) # 使用余弦相似度作为度量
    print("similarity shape", similarity.shape)
    return similarity

def build_graph(graph, prompts, similarity):
    print("similarity", similarity)
    for i, prompt in enumerate(prompts):
        graph.add_node(i, label=prompt) # 添加节点，用序号作为标识，用prompt作为标签
    for i in range(len(prompts)):
        for j in range(i+1, len(prompts)):
            graph.add_edge(i, j, weight=similarity[i][j]) # 添加边，用相似度作为权重
    fig_name = "prompts_graph"

    fig, ax = plt.subplots(figsize=(3, 3))


    pos = nx.spring_layout(graph, weight='weight')

    nx.draw_networkx(graph, ax=ax,pos = pos, node_color="lightblue", edge_color="gray", font_size=1, font_weight="bold",
                     node_size= 1)

    # # 调整坐标轴的边界
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)

    # 隐藏坐标轴的刻度
    ax.set_xticks([])
    ax.set_yticks([])
    #nx.draw(graph, with_labels=False, node_color="lightblue", edge_color="gray", font_size=10, font_weight="bold")
    plt.savefig('figs/{}.pdf'.format(fig_name), bbox_inches='tight')
    return graph


def get_influence(graph):
    communities = nx.algorithms.community.greedy_modularity_communities(graph) # 使用贪心算法划分社区
    influence = {} # 用来存储节点的影响力
    for i, community in enumerate(communities):
        print("community", i, community)
        subgraph = graph.subgraph(community) # 获取子图
        centrality = nx.degree_centrality(subgraph) # 计算子图中节点的度中心性
        for node, value in centrality.items():
            influence[node] = value # 将度中心性作为节点的影响力
    return influence

def plot_embedding(goals_embeddings, n_clusters = 2,fig_name = "goal_prompts"):
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(goals_embeddings)
    # 将嵌入向量转换为2D空间，以便于可视化
    tsne = TSNE(n_components=2, random_state=0)
    embeddings_2d = tsne.fit_transform(goals_embeddings)

    # 绘制所有的嵌入向量，并根据其聚类标签着色
    plt.figure(figsize=(10, 10))
    for i in range(n_clusters):
        # 选择属于某个聚类的点
        indices = kmeans.labels_ == i
        plt.scatter(embeddings_2d[indices, 0], embeddings_2d[indices, 1], label=f'Cluster {i}')
    plt.legend()
    plt.title('t-SNE visualization of the embeddings clustered by K-Means')
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.savefig('figs/{}.pdf'.format(fig_name), bbox_inches='tight')
    plt.show()
    plt.close()
##########################################################
def get_adv_suffix(file_path: str):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    adv_suffixs = []

    for item in data:
        print("item",item)
        if "user_prompt" in item:
            adv_suffixs.append(item["user_prompt"] + "" + item["adv_prompt"])
        else:
            adv_suffixs.append(item["original_prompt"] + "" + item["adv_prompt"])
    return adv_suffixs

########################################################
def get_adv_prompts(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    adv_prompts = []
    for item in data:
        adv_prompts.append(item["adv_prompt"])

    return adv_prompts


