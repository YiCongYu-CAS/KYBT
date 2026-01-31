
% （1）先用 subplot 自动布局，记下每个axes位置
fig = fig88;
set(fig, 'Units', 'pixels');
set(fig, 'Position', [200, 0, 850, 750]);  % 先用正常高度


axesList = [ax1, ax2, ax3, ax4];
axesPos = cell(1,4);
for k = 1:4
    set(axesList(k), 'Units', 'pixels');                % 变成像素单位
    axesPos{k} = get(axesList(k), 'Position');          % 记下原始subplot位置
end

drawnow; % 保证刷新

% （2）把窗口高度调大，产生顶部留白
set(fig, 'Position', [200, 0, 850, 765]);  % 高度加大！现在会出现大留白

% （3）用像素单位把每个 axes 恢复到原来的大小和位置
for k = 1:4
    set(axesList(k), 'Units', 'pixels');
    set(axesList(k), 'Position', axesPos{k});  % 强行锁死回原位置！
end

% （4）加 legend 到顶部
lgd = legend([h1, h2, h3, h4], {'$N_m=3252$ ', '$N_m=5280$ ', '$N_m=9456$ ', '$N_m=21600$ '}, 'Interpreter', 'latex');
% 设置图例为横向排列
lgd.Orientation = 'horizontal';
% 去掉图例边框
lgd.Box = 'off';

lgd.Position = [0.3, 0.95, 0.1, 0.1]; % 自定义位置
set(lgd, 'Units', 'pixels');
figPos = get(fig, 'Position');
lgdPos = get(lgd, 'Position');
lgdLeft = (figPos(3) - lgdPos(3))/2;
lgdBottom = figPos(4) - lgdPos(4) - 30; % 30像素下移
set(lgd, 'Position', [lgdLeft, lgdBottom, lgdPos(3), lgdPos(4)]);

title(ax1, '$\Gamma_1$', 'Interpreter', 'latex', 'FontSize', 24);
title(ax2, '$\Gamma_2$', 'Interpreter', 'latex', 'FontSize', 24);
title(ax3, '$\Gamma_4$', 'Interpreter', 'latex', 'FontSize', 24);
title(ax4, '$\Gamma_5^{(1)}$', 'Interpreter', 'latex', 'FontSize', 24);
set(fig, 'Color', 'white');

for k = 1:4
    set(get(axesList(k), 'Title'), 'Units', 'normalized', 'Position', [0.5, 0.9, 0], 'VerticalAlignment', 'bottom');
end


set(h1, 'color', co1, 'linewidth', myLineWidth/2, 'MarkerSize',myMarkerSize, 'MarkerFaceColor',co1in);
set(h2, 'color', co2, 'linewidth', myLineWidth/2, 'MarkerSize',myMarkerSize, 'MarkerFaceColor',co2in);
set(h3, 'color', co6, 'linewidth', myLineWidth/2, 'MarkerSize',myMarkerSize, 'MarkerFaceColor',co6in);
set(h4, 'color', co5, 'linewidth', myLineWidth/2, 'MarkerSize',myMarkerSize, 'MarkerFaceColor',co5in);

% 找到所有线对象
h = findobj(tf, 'Type', 'line'); 
% 初始化最大值
max_y = -inf;
% 遍历所有线对象
for i = 1:length(h)
    y_data = get(h(i), 'YData'); % 获取每条线的 Y 数据
    max_y = max(max_y, max(y_data)); % 更新最大值
end

% tf.YAxis.Label.Position(1) = tf.YAxis.Label.Position(1) -0;

% tf.YAxis.TickLabelInterpreter = 'latex'; % 切换为 LaTeX 解释器

% tf.YAxis.TickLabelFormat = '$10^{%+02d}$'; % 格式化为 LaTeX 的指数形式

ylim(tf, [0, max_y*1.25]);
title(tf, '$\Gamma_1$', 'Interpreter', 'latex', 'FontSize', 24);

set(fig88, 'Position',  [100, 0, 800, 700]);  % 设置图形位置和大小
set(fig88, 'Color', 'white');
set(fig88, 'OuterPosition', [100, 0, 900, 1000]);

lgd = legend([h1, h2, h3, h4], {'$N_m=3252$', '$N_m=3252$', '$N_m=3252$', '$N_m=3252$'}, 'Interpreter', 'latex');
% 设置图例为横向排列
lgd.Orientation = 'horizontal';
% 去掉图例边框
lgd.Box = 'off';

lgd.Position = [0.3, 0.95, 0.1, 0.1]; % 自定义位置

ax = subplot(2,2,2);  % 得到第1个子图句柄
set(ax, 'Units', 'pixels');   % 设置为像素单位
pos = get(ax, 'Position');    % 取出位置和大小 [left bottom width height]
pos(4) = pos(4) * 0.8;        % 高度缩成原来一半（你可以随便改成你想要的像素值，比如120）
set(ax, 'Position', pos);     % 设置回去