function plot_fem_basis_hat_demo()
% 演示：PDE Toolbox 生成的三角形网格
% 左：二维网格；右：在某个节点为1、其余为0的线性基函数（帽函数）
% 兼容 MATLAB R2017b（使用 initmesh/pdemesh/trisurf/print）

    close all; clc;
    
   

    % 1) 定义一个简单几何区域（单位正方形）
    % PDE Toolbox 需要以 polygon 形式输入：矩形 [0,1]x[0,1]
    gd = [3;4; 0;1;1;0; 0;0;1;1];  % [type; nV; x1..x4; y1..y4], type=3->polygon
    ns = char('R1')';               % 名称
    sf = 'R1';                      % set formula
    [dl,bt] = decsg(gd, sf, ns); %#ok<ASGLU>

    % 2) 生成较稀的三角形网格（通过 hmax 控制）
    hmax = 0.3;     % 网格尺寸，数值越大网格越稀，可调为 0.3/0.35 等
    [p,e,t] = initmesh(dl, 'hmax', hmax);  % p: 2xN 节点; t: 4xNT（三角形的点索引在 t(1:3,:)）

    % 3) 选择一个“激活节点”作为基函数中心（也可手动指定）
    % 这里选最靠近中心 (0.5,0.5) 的网格节点
    pts = p';           % N x 2
    [~,activeNode] = min(sum((pts - repmat([0.7,0.5], size(pts,1), 1)).^2, 2));
    % 也可手动设定，比如 activeNode = 10;

    % 4) 准备节点值 u（帽函数：选中节点=1，其它=0）
    N = size(p,2);
    u = zeros(N,1);
    u(activeNode) = 1;

    % 5) 绘图：左边二维网格，右边三维帽函数
    figure('Color','w','Position',[100,100,1000,420]);

    % 左图：二维网格
    subplot(1,2,1); hold on;
    pdemesh(p, e, t, 'NodeLabels', 'on', 'ElementLabels', 'off');
    % 高亮激活节点
    plot(p(1,activeNode), p(2,activeNode), 'ro', 'MarkerFaceColor', 'r', 'MarkerSize', 10);
   
    axis equal tight; box on;
    xlabel('x','fontname', 'Times New Roman', 'fontsize', 20); 
    ylabel('y','fontname', 'Times New Roman', 'fontsize', 20);
    set(gca, 'FontSize', 20);
    
     title('The constructed  mesh','fontname', 'Times New Roman','fontsize', 20);
     
    
     
     

    % 右图：三维帽函数（在网格上线性插值）
    subplot(1,2,2); hold on;
    tri = t(1:3,:)';                  % NT x 3 的三角形连接
    hsurf = trisurf(tri, p(1,:)', p(2,:)', u);
    set(hsurf, 'FaceColor', 'interp', 'EdgeColor', [0.4 0.4 0.4], 'FaceAlpha', 0.8);
%     colormap(parula(256));
 colormap(cool(256));
    caxis([0 1]); zlim([0 1]);

    % 右图中高亮激活节点
    scatter3(p(1,activeNode), p(2,activeNode), 1, 36, 'r', 'filled');
    text(p(1,activeNode), p(2,activeNode), 1.05, ' ', ...
        'HorizontalAlignment','center','Color','r');

    % 右图外观
    grid on; box on; axis equal tight;
    view(30,20);
    
    xlabel('x','fontname', 'Times New Roman', 'fontsize', 20); 
    ylabel('y','fontname', 'Times New Roman', 'fontsize', 20);
    zlabel('\phi_{23}(x,y)','fontname', 'Times New Roman', 'fontsize', 20 );
    set(gca, 'FontSize', 20)
    title('The basis corresponding to a point','fontname', 'Times New Roman', 'fontsize', 20 );
    
    
     h1 = subplot(1,2,1);
h2 = subplot(1,2,2);
    
   

% 设置为合适的位置，保证不会丢失标签
set(h1, 'Position', [0.0 0.2 0.5 0.7]);
set(h2, 'Position', [0.5 0.2 0.5 0.7]);
 % 获取位置向量 [left bottom width height]
pos1 = get(h1, 'Position');
pos2 = get(h2, 'Position');
% 手动微调
pos1(1) = pos1(1) + 0.0;
pos1(2) = pos1(2) + 0.0;
pos1(3) = pos1(3) - 0;  % 缩小宽度
set(h1, 'Position', pos1);

pos2(1) = pos2(1) + 0.0;  % 向右移动
pos2(3) = pos2(3) - 0;  % 缩小宽度
set(h2, 'Position', pos2);



    % 可选：导出图片（R2017b 建议 print 或 saveas）
    % print(gcf, '-dpng', '-r200', 'fem_basis_hat_demo.png');
   fig = gcf;
set(fig, 'Units', 'centimeters');
fig_pos = get(fig, 'Position');   % [left, bottom, width, height]
set(fig, 'PaperUnits', 'centimeters');
set(fig, 'PaperPosition', [0 0 fig_pos(3) fig_pos(4)]);
set(fig, 'PaperSize', [fig_pos(3) fig_pos(4)]);
 print(gcf, 'FigSchematicFEM01.pdf', '-dpdf');   
    
    
end