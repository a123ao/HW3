# HW3-4 加分題：使用 Rainbow DQN 解 Random Mode GridWorld

這份文件將帶你先「分析」Rainbow DQN 的核心組成，然後「教你怎麼做」，一步步將這些技術應用在隨機模式的 GridWorld 環境中。

## 一、Rainbow DQN 核心分析 🌈

DeepMind 在 2017 年提出的 **Rainbow DQN**，顧名思義是結合了當時 DQN 的 6 種最有效擴充技術。GridWorld 的 Random Mode 是最困難的模式，因為所有物件位置都不固定，傳統 DQN 很難有效率地探索並收斂。導入 Rainbow 可以大幅穩定並加速訓練：

1. **Double Q-Learning (雙重 Q 學習)**
   - **解決問題**：傳統 DQN 容易「高估」Q 值。
   - **作法**：使用 Online Network 來選擇動作，再用 Target Network 來計算該動作的 Q 值。我們在 HW3-2 已經實作過。

2. **Dueling Networks (競爭網路架構)**
   - **解決問題**：在很多狀態下，採取什麼動作其實不重要（例如離目標還很遠時）。
   - **作法**：將網路分成兩支，分別預測 State Value $V(s)$ 與 Advantage $A(s, a)$。我們在 HW3-2 也實作過。

3. **Prioritized Experience Replay, PER (優先經驗回放)**
   - **解決問題**：標準的 Replay Buffer 是隨機抽樣，但有些經驗（例如出乎意料的失敗或成功）更能讓模型學到東西。
   - **作法**：根據 TD-Error (預測值與目標值的誤差) 的大小給予經驗不同的權重，誤差越大的經驗越容易被抽中。

4. **Multi-step Learning (多步學習, N-step Returns)**
   - **解決問題**：單步的 Reward 傳遞太慢。
   - **作法**：不只看下一步的 Reward，而是看未來 N 步的 Reward 總和來更新目前的 Q 值。

5. **Distributional RL (分佈式強化學習)**
   - **解決問題**：傳統 DQN 只預測 Q 值的「期望值」，但這無法呈現風險或變異。
   - **作法**：網路改為輸出一個機率分佈 (Value Distribution)。

6. **Noisy Nets (雜訊網路)**
   - **解決問題**：$\epsilon$-greedy 的探索策略太過隨機且死板。
   - **作法**：在神經網路的線性層權重中加入可學習的 Gaussian Noise，讓網路自然而然地具備探索能力。

---

## 二、教你怎麼做：實戰 Mini-Rainbow

要在作業中從零刻出完整的 Rainbow (包含 Distributional 網路) 非常複雜，程式碼可能會超過幾百行。因此，實務上我們通常會實作 **"Mini-Rainbow"**，結合投資報酬率最高的 4 個技巧：**Double + Dueling + Multi-step + PER**。

以下是教你如何改造目前的 DQN 成為 Mini-Rainbow 的步驟與關鍵程式碼片段：

### 步驟 1：實作 N-step Buffer
當我們把資料存進 Buffer 時，不要馬上存 `(s, a, r, s')`，而是維護一個長度為 $N$ 的暫存佇列。
當佇列滿了，計算 N 步的累積 Reward：
$R^{(n)} = r_t + \gamma r_{t+1} + ... + \gamma^{n-1} r_{t+n-1}$
然後將 `(s_t, a_t, R^{(n)}, s_{t+n})` 存進 Buffer。

```python
# N-step 暫存區的簡易實作概念
n_step_buffer = deque(maxlen=3) # 以 3-step 為例

def push_n_step(state, action, reward, next_state, done):
    n_step_buffer.append((state, action, reward, next_state, done))
    if len(n_step_buffer) == 3:
        # 計算 3 步累積 reward
        s, a, _, _, _ = n_step_buffer[0]
        _, _, _, s_next, d = n_step_buffer[-1]
        
        R = 0
        for i, transition in enumerate(n_step_buffer):
            R += transition[2] * (gamma ** i)
            if transition[4]: # 如果中途結束了
                break
        
        # 存進真正的 Replay Buffer
        real_buffer.push(s, a, R, s_next, d)
```

### 步驟 2：實作 PER (優先經驗回放)
你需要一個可以根據機率抽樣的 Buffer。簡單的做法是儲存每個 transition 的 `priority = |TD_error| + epsilon`。
- 每次抽樣時，機率 $P(i) = p_i^\alpha / \sum p_k^\alpha$
- 為了抵消抽樣機率改變帶來的偏差，計算 Loss 時要乘上 Importance Sampling Weight: $w_i = (N \cdot P(i))^{-\beta}$

### 步驟 3：結合 Double 與 Dueling (我們 HW3-2 已完成)
將 `hw3_2_variants_player.py` 中的 `DuelingDQN` 網路拿來用，並且在計算 Loss 時使用 Double DQN 的邏輯：
```python
# 結合 Dueling 網路結構
q_values = model(states) 
q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

# Double DQN 選擇動作與評估
with torch.no_grad():
    online_next_q_values = model(next_states)
    next_actions = torch.argmax(online_next_q_values, dim=1)
    
    target_next_q_values = target_model(next_states)
    next_q_value = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)

# 注意這裡的 discount factor 要變成 gamma ** N
expected_q_value = rewards + (gamma ** N) * next_q_value * (1 - dones)
```

### 步驟 4：套用到 Random Mode
隨機模式因為每次起點和終點都不同，需要**更大的 Buffer** 和**更長的探索期 (Epsilon decay 放慢)**。
使用 Mini-Rainbow 後，PER 會幫助模型自動挑選「恰好走到終點或陷阱」的稀有經驗來頻繁訓練，Multi-step 會把得到 Reward 的訊號快速往前傳遞，這兩個技巧能最直接解決 Random Mode 難以訓練的問題！

> **建議**：你可以將 HW3-3 的 PyTorch Lightning 程式碼複製一份，先加入 Dueling 和 Double 邏輯，然後實作一個簡單的 N-step deque。如果你想挑戰 PER，可以引用社群開源的 `SegmentTree` 來實作有效率的 PER Buffer。
