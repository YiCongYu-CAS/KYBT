function plot_single_triangle_hat3D()
% 单个三角形上的线性基函数（帽函数）教学示意
% 一张三维图：xy 平面三角形 + 上方多棱锥曲面 + 从 P1 到抬起点的连线
% 兼容 MATLAB R2017b

    close all; clc;

    % 三角形顶点（可改为你的实际三角形）
    P = [0 0; 1.2 0.1; 0.35 1.1];   % P1, P2, P3
    names = {'P1','P2','P3'};

    % 选择抬起的顶点（1/2/3）
    activeVertex = 1;

    % 参考三角形采样并仿射映射到实际三角形
    n = 120; % 采样密度（示意平滑）
    [xr, yr, mask] = ref_triangle_grid(n);
    X = P(1,1) + (P(2,1)-P(1,1))*xr + (P(3,1)-P(1,1))*yr;
    Y = P(1,2) + (P(2,2)-P(1,2))*xr + (P(3,2)-P(1,2))*yr;

    % 线性形函数（参考域）
    N1 = 1 - xr - yr; N2 = xr; N3 = yr;
    Nall = cat(3, N1, N2, N3);
    Z = Nall(:,:,activeVertex);
    Z(~mask) = NaN;

    % 绘图
    figure('Color','w','Position',[80,80,780,620]); hold on;

    % 1) xy 平面三角形（z=0）
    plot3([P(:,1); P(1,1)], [P(:,2); P(1,2)], [0;0;0;0], ...
          'Color',[0.4 0.4 0.4], 'LineWidth',1.5);
    for k=1:3
        scatter3(P(k,1), P(k,2), 0, 36, 'k', 'filled');
        text(P(k,1), P(k,2), -0.05, names{k}, ...
            'HorizontalAlignment','center','VerticalAlignment','top', 'Color','k');
    end

    % 2) 三维“多棱锥”曲面
    hs = surf(X, Y, Z, Z, 'EdgeColor','none', 'FaceColor','interp', 'FaceAlpha',0.6); %#ok<NASGU>
    colormap(cool(256)); caxis([0 1]);

    % 3) 等值线
    try
        [~,hc] = contour3(X, Y, Z, 8, 'k:'); %#ok<ASGLU>
        set(hc,'LineWidth',0.8);
    catch
    end

    % 4) 高亮抬起的顶点（z=1）
    Pv = P(activeVertex,:);
    scatter3(Pv(1), Pv(2), 1, 60, [0.85 0 0], 'filled');
    text(Pv(1), Pv(2), 1.06, sprintf('%s: 1', names{activeVertex}), ...
        'HorizontalAlignment','center','Color',[0.85 0 0]);

    % 5) 从 P1 到其抬起点的连线（教学强调）
    %    仅当 activeVertex==1 时，这条线正好指向峰值
    P1 = P(1,:);
    plot3([P1(1) P1(1)], [P1(2) P1(2)], [0 1], ...
          '--', 'Color', [0 0 0], 'LineWidth', 2.0);
    % 在底端与顶端各加一个小标注点
    scatter3(P1(1), P1(2), 0, 28, [0.85 0 0], 'filled');
    scatter3(P1(1), P1(2), 1, 28, [0.85 0 0], 'filled');

    % 可选：如果你希望无论哪个顶点被抬起，都画对应顶点的连线，
    % 可改为：把上面 P1 替换为 P(activeVertex,:)。

    % 轴与视图（范围稍放大）
    axis equal tight;
    pad = 0.2;
    xlim([min(P(:,1))-pad, max(P(:,1))+pad]);
    ylim([min(P(:,2))-pad, max(P(:,2))+pad]);
    zlim([-0.1, 1.05]);
    grid on; box on;
    view(36,28);
    xlabel('x'); ylabel('y'); zlabel(sprintf('N_{%d}(x,y)', activeVertex));
    title(sprintf('单个三角形上 %s 抬起的线性基函数', names{activeVertex}));

    % 导出（可选）
    % print(gcf, '-dpng', '-r220', sprintf('single_tri_hat_P%d.png', activeVertex));
end

function [xr, yr, mask] = ref_triangle_grid(n)
% 参考三角形 Tref={(x,y)| x>=0, y>=0, x+y<=1 } 上的规则网格
    [xr, yr] = meshgrid(linspace(0,1,n), linspace(0,1,n));
    mask = (xr>=0) & (yr>=0) & (xr + yr <= 1);
    xr(~mask) = NaN; yr(~mask) = NaN;
end