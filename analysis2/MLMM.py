import os
import json
import fixation_data_utils as f_utils
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from scipy.stats import norm  # 用于计算 p 值

np.random.seed(123)
torch.manual_seed(123)

# 检查是否有可用的 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

data_file = 'Data/sketches/'
sketch_svgs = [sketch_svg for sketch_svg in os.listdir(data_file) if sketch_svg.endswith(".svg")]
with open('./Data/registed_eyetrack_drawing_with_fixation_point.json', 'r') as json_file:
    eyetrack_and_drawing_data = json.load(json_file)

user_data = pd.DataFrame(columns=['uid', 'image', 'x1', 'y1', 'x2', 'y2'])

for sketch_svg in sketch_svgs:
    image = sketch_svg[:-15] + '.png'
    uid = sketch_svg[-14:-4]
    n_uid = int(uid[-2:])
    # print(n_uid)
    # for paper
    # if sketch_svg != "DIY_Gantry_bust_010_RGB_eyetrack09.svg" and sketch_svg != "DIY_Gantry_teapot_020_RGB_eyetrack04.svg":
    #     continue
    
    fixation_datas = eyetrack_and_drawing_data[uid][image]
    combined_fixation_datas = f_utils.combine_fixation_points(fixation_datas)
    
    fixation_centers = [fixation_data.center for fixation_data in combined_fixation_datas]
    
    fixation_time = [np.array([fixation_data.startTime, fixation_data.endTime]) for fixation_data in combined_fixation_datas]
    if (len(fixation_time) == 0):
        continue
    fixation_time[len(fixation_time)-1][1] = fixation_time[len(fixation_time)-1][0]
    fixation_time[0][0] = fixation_time[0][1]
    # print(np.min(fixation_time))
    fixation_time = (fixation_time - np.min(fixation_time)) / (np.max(fixation_time) - np.min(fixation_time))

    strokes_centers = [fixation_data.getPointCenter() for fixation_data in combined_fixation_datas]
    original_strokes_centers = strokes_centers.copy()
    #strokes_time = [fixation_data.getStrokesTime() for fixation_data in combined_fixation_datas]
    #strokes_time = (strokes_time - np.min(strokes_time)) / (np.max(strokes_time) - np.min(strokes_time))
    
    #max_time = np.max(strokes_time)
    #fixation_time = (fixation_time - np.min(fixation_time)) / (max_time - np.min(fixation_time))
    #strokes_time = (strokes_time - np.min(strokes_time)) / (max_time - np.min(strokes_time))
    # print(image)
    fixation_centers = np.array(fixation_centers)
    strokes_centers = np.array(strokes_centers)
    for i in range(len(strokes_centers)):
        user_data.loc[len(user_data)] = [n_uid, image, fixation_centers[i][0], fixation_centers[i][1], strokes_centers[i][0], strokes_centers[i][1]]

# group py artists
features = {
    'x1': torch.tensor(user_data['x1'].values, dtype=torch.float32).to(device),
    'y1': torch.tensor(user_data['y1'].values, dtype=torch.float32).to(device),
    'group': torch.tensor(user_data['uid'].values, dtype=torch.long).to(device)
}

observations = torch.tensor(user_data[['x2', 'y2']].values, dtype=torch.float32).to(device)

class MultivariateLinearMixedEffectModel(nn.Module):
    def __init__(self, num_groups):
        super(MultivariateLinearMixedEffectModel, self).__init__()

        self.intercept = nn.Parameter(torch.zeros(2, device=device))  
        self.effect_x1 = nn.Parameter(torch.zeros(2, device=device))  
        self.effect_y1 = nn.Parameter(torch.zeros(2, device=device))  

        self.group_effects = nn.Embedding(num_groups, 2).to(device)  
        self.group_effects.weight.data.normal_(mean=0, std=0.1)  

    def forward(self, x1, y1, group):
        fixed_effects = self.intercept + self.effect_x1 * x1.unsqueeze(1) + self.effect_y1 * y1.unsqueeze(1)

        random_effects = self.group_effects(group)

        total_effects = fixed_effects + random_effects
        return total_effects

num_groups = user_data['uid'].nunique()
print("num_groups:", num_groups)
model = MultivariateLinearMixedEffectModel(num_groups).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 10000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(features['x1'], features['y1'], features['group'])
    loss = criterion(outputs, observations)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

print("\n--- Fix effect parameter---")
print("Intercept:", model.intercept.data.cpu().numpy())
print("Effect of x1:", model.effect_x1.data.cpu().numpy())
print("Effect of y1:", model.effect_y1.data.cpu().numpy())

group_effects = model.group_effects.weight.data.numpy()
print("Group Effects (Random Effects):", group_effects)

def calculate_statistics(model, observations):

    fixed_effects = [model.intercept, model.effect_x1, model.effect_y1]
    fixed_names = ["Intercept", "Effect of x1", "Effect of y1"]
    
    optimizer.zero_grad()
    outputs = model(features['x1'], features['y1'], features['group'])
    loss = criterion(outputs, observations)
    loss.backward()

    print("\n--- Fix effect report ---")
    print(f"{'Parameter':<20} {'Estimate':<10} {'t-value':<10} {'p-value':<10}")
    for param, name in zip(fixed_effects, fixed_names):
        estimate = param.data.cpu().numpy()
        std_err = param.grad.std().cpu().numpy()  
        t_value = estimate / std_err
        p_value = 2 * (1 - norm.cdf(np.abs(t_value)))  
        print(f"{name:<20} {estimate[0]:<10.4f} {t_value[0]:<10.4f} {p_value[0]:<10.4f}")
        print(f"{name:<20} {estimate[1]:<10.4f} {t_value[1]:<10.4f} {p_value[1]:<10.4f}")

calculate_statistics(model, observations)