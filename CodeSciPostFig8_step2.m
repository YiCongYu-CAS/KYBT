set(0, 'DefaultAxesFontName', 'Times New Roman');  % 设置轴的默认字体
set(0, 'DefaultTextFontName', 'Times New Roman');  % 设置文本的默认字体
set(0, 'DefaultAxesFontSize', 16);  % 设置默认的坐标轴字体大小为20

co1 = addcolor(113);
co1in = addcolor(107);

co2 = addcolor(139);
co2in = addcolor(141);

co3 = addcolor(198);
co3in = addcolor(211);

co4 = addcolor(129);
co4in = addcolor(133);

co5 = addcolor(166);
co5in = addcolor(165);

co6 = addcolor(74);
co6in = addcolor(65);

vx = 1:20;
vy = vEnTotal(1:20);
vt = vType(1:20);

close(figure(1));
tf = figure(1);
set(tf, 'Position',  [300, -200, 900, 850]);  % 设置图形位置和大小
set(tf, 'Color', 'white');

subplot(3,3,1);
hold on;

h0 = plot(vx, vy, '-black');

for ix = 1:length(vx)
    
   x = vx(ix);
   y = vEnTotal(ix);
   tp = vType(ix);
   
   if tp == 1
       h1 = plot(x, y, 'o'); 
       set(h1, 'color', co1, 'linewidth', 1, 'MarkerSize',6, 'MarkerFaceColor',co1in);
   elseif tp == 2
       h2 = plot(x, y, '^'); 
       set(h2, 'color', co2, 'linewidth', 1, 'MarkerSize',6, 'MarkerFaceColor',co2in);
   elseif tp == 3
       h3 = plot(x, y, 'v'); 
       set(h3, 'color', co3, 'linewidth', 1, 'MarkerSize',6, 'MarkerFaceColor',co3in);
   elseif tp == 4
       h4 = plot(x, y, 's'); 
       set(h4, 'color', co4, 'linewidth', 1, 'MarkerSize',6, 'MarkerFaceColor',co4in);
   elseif (tp == 51) || (tp == 52)
       h5 = plot(x, y, 'd'); 
       set(h5, 'color', co5, 'linewidth', 1, 'MarkerSize',6, 'MarkerFaceColor',co5in);
   elseif (tp == 61) || (tp == 62)
       h6 = plot(x, y, 'h'); 
       set(h6, 'color', co6, 'linewidth', 1, 'MarkerSize',6, 'MarkerFaceColor',co6in);
   end
    
end

xlim([0, 20.5]); % 设置 y 轴范围
ylim([0, 200]); % 设置 y 轴范围
xlabel('$i$', 'Interpreter', 'latex', 'FontSize', 16);
ylabel('$\epsilon_i$', 'Interpreter', 'latex', 'FontSize', 16);
title('Spectrum', 'FontSize', 16,'FontName','Times New Roman');
set(gca, 'YTick', [0,100,200]);

hLegend = legend([h1, h2, h3, h4, h5, h6], ...
    {'$\Gamma_1$', '$\Gamma_2$', '$\Gamma_3$','$\Gamma_4$','$\Gamma_5$','$\Gamma_6$'}, ...
    'interpreter', 'latex','fontsize',10);
set(hLegend, 'Position', [0.125, 0.81, 0.08, 0.08]); 
% 关闭图例框
set(hLegend, 'Box', 'off');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,3,2);

cSolution = cWaveFuncGamma1{1};
vu = fn_Ext(p0, p, cSolution, cgMatrix);
[~, ind] = max(abs(vu));
vu = vu/vu(ind);

pdeplot(p, [], t, 'XYData', vu, 'Contour', 'off', 'Mesh', 'off');
colormap('jet');  % 设置颜色映射
caxis([0, 1]);
tmp = colorbar;   % 添加颜色条
% 设置颜色条刻度和刻度标签
tmp.Ticks = [-1,-0.5, 0, 0.5, 1];        % 设置刻度位置
% tmp.TickLabels = {'Low', 'Mid', 'High'};  % 设置刻度标签
title('$\Gamma_1$ GS', 'interpreter', 'latex', 'FontSize', 16);
xlabel('$x$',  'interpreter', 'latex');
ylabel('$y$',  'interpreter', 'latex');
view(2);  % 设置为三维视角
xlim([-1.04, 1.02]); % 设置 y 轴范围
ylim([-0.9, 0.9]); % 设置 y 轴范围
% axis equal;  % 坐标轴等比例显示

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,3,3);

cSolution = cWaveFuncGamma2{1};
vu = fn_Ext(p0, p, cSolution, cgMatrix);
[~, ind] = max(abs(vu));
vu = vu/vu(ind);

pdeplot(p, [], t, 'XYData', vu, 'Contour', 'off', 'Mesh', 'off');
colormap('jet');  % 设置颜色映射
colorbar;  % 显示颜色条
caxis([-1, 1]);
tmp = colorbar;   % 添加颜色条
% 设置颜色条刻度和刻度标签
tmp.Ticks = [-1,-0.5, 0, 0.5, 1];        % 设置刻度位置
% tmp.TickLabels = {'Low', 'Mid', 'High'};  % 设置刻度标签
title('$\Gamma_2$ GS', 'interpreter', 'latex', 'FontSize', 16);
xlabel('$x$',  'interpreter', 'latex');
ylabel('$y$',  'interpreter', 'latex');
view(2);  % 设置为三维视角
xlim([-1.04, 1.02]); % 设置 y 轴范围
ylim([-0.9, 0.9]); % 设置 y 轴范围
% axis equal;  % 坐标轴等比例显示

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,3,4);

cSolution = cWaveFuncGamma3{1};
vu = fn_Ext(p0, p, cSolution, cgMatrix);
[~, ind] = max(abs(vu));
vu = vu/vu(ind);


pdeplot(p, [], t, 'XYData', vu, 'Contour', 'off', 'Mesh', 'off');
colormap('jet');  % 设置颜色映射
colorbar  % 显示颜色条
caxis([-1, 1]);
tmp = colorbar;   % 添加颜色条
% 设置颜色条刻度和刻度标签
tmp.Ticks = [-1,-0.5, 0, 0.5, 1];        % 设置刻度位置
% tmp.TickLabels = {'Low', 'Mid', 'High'};  % 设置刻度标签
title('$\Gamma_3$ GS', 'interpreter', 'latex', 'FontSize', 16);
xlabel('$x$',  'interpreter', 'latex');
ylabel('$y$',  'interpreter', 'latex');
view(2);  % 设置为三维视角
xlim([-1.04, 1.02]); % 设置 y 轴范围
ylim([-0.9, 0.9]); % 设置 y 轴范围
% axis equal;  % 坐标轴等比例显示

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,3,5);

cSolution = cWaveFuncGamma4{1};
vu = fn_Ext(p0, p, cSolution, cgMatrix);
[~, ind] = max(abs(vu));
vu = vu/vu(ind);


pdeplot(p, [], t, 'XYData', vu, 'Contour', 'off', 'Mesh', 'off');
colormap('jet');  % 设置颜色映射
colorbar  % 显示颜色条
caxis([-1, 1]);
tmp = colorbar;   % 添加颜色条
% 设置颜色条刻度和刻度标签
tmp.Ticks = [-1,-0.5, 0, 0.5, 1];        % 设置刻度位置
% tmp.TickLabels = {'Low', 'Mid', 'High'};  % 设置刻度标签
title('$\Gamma_4$ GS', 'interpreter', 'latex', 'FontSize', 16);
xlabel('$x$',  'interpreter', 'latex');
ylabel('$y$',  'interpreter', 'latex');
view(2);  % 设置为三维视角
xlim([-1.04, 1.02]); % 设置 y 轴范围
ylim([-0.9, 0.9]); % 设置 y 轴范围
% axis equal;  % 坐标轴等比例显示

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,3,6);

cSolution = cWaveFuncGamma5_1{1};
vu = fn_Ext(p0, p, cSolution, cgMatrix);
[~, ind] = max(abs(vu));
vu = vu/vu(ind);


pdeplot(p, [], t, 'XYData', vu, 'Contour', 'off', 'Mesh', 'off');
colormap('jet');  % 设置颜色映射
colorbar;  % 显示颜色条
caxis([-1, 1]);
tmp = colorbar;   % 添加颜色条
% 设置颜色条刻度和刻度标签
tmp.Ticks = [-1,-0.5, 0, 0.5, 1];        % 设置刻度位置
% tmp.TickLabels = {'Low', 'Mid', 'High'};  % 设置刻度标签
title('$\Gamma_5^{(1)}$ GS', 'interpreter', 'latex', 'FontSize', 16);
xlabel('$x$',  'interpreter', 'latex');
ylabel('$y$',  'interpreter', 'latex');
view(2);  % 设置为三维视角
xlim([-1.04, 1.02]); % 设置 y 轴范围
ylim([-0.9, 0.9]); % 设置 y 轴范围
% axis equal;  % 坐标轴等比例显示

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,3,7);

cSolution = cWaveFuncGamma5_2{1};
vu = fn_Ext(p0, p, cSolution, cgMatrix);
[~, ind] = max(abs(vu));
vu = vu/vu(ind);


pdeplot(p, [], t, 'XYData', vu, 'Contour', 'off', 'Mesh', 'off');
colormap('jet');  % 设置颜色映射
colorbar  % 显示颜色条
caxis([-1, 1]);
tmp = colorbar;   % 添加颜色条
% 设置颜色条刻度和刻度标签
tmp.Ticks = [-1,-0.5, 0, 0.5, 1];        % 设置刻度位置
% tmp.TickLabels = {'Low', 'Mid', 'High'};  % 设置刻度标签
title('$\Gamma_5^{(2)}$ GS', 'interpreter', 'latex', 'FontSize', 16);
xlabel('$x$',  'interpreter', 'latex');
ylabel('$y$',  'interpreter', 'latex');
view(2);  % 设置为三维视角
xlim([-1.04, 1.02]); % 设置 y 轴范围
ylim([-0.9, 0.9]); % 设置 y 轴范围
% axis equal;  % 坐标轴等比例显示

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,3,8);

cSolution = cWaveFuncGamma6_1{1};
vu = fn_Ext(p0, p, cSolution, cgMatrix);
[~, ind] = max(abs(vu));
vu = vu/vu(ind);


pdeplot(p, [], t, 'XYData', vu, 'Contour', 'off', 'Mesh', 'off');

colormap('jet');  % 设置颜色映射
caxis([-1, 1]);
tmp = colorbar;   % 添加颜色条
% 设置颜色条刻度和刻度标签
tmp.Ticks = [-1,-0.5, 0, 0.5, 1];        % 设置刻度位置
% tmp.TickLabels = {'Low', 'Mid', 'High'};  % 设置刻度标签


title('$\Gamma_6^{(1)}$ GS', 'interpreter', 'latex', 'FontSize', 16);
xlabel('$x$',  'interpreter', 'latex');
ylabel('$y$',  'interpreter', 'latex');
view(2);  % 设置为三维视角
xlim([-1.04, 1.02]); % 设置 y 轴范围
ylim([-0.9, 0.9]); % 设置 y 轴范围
% axis equal;  % 坐标轴等比例显示

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

subplot(3,3,9);

cSolution = cWaveFuncGamma6_2{1};
vu = fn_Ext(p0, p, cSolution, cgMatrix);
[~, ind] = max(abs(vu));
vu = vu/vu(ind);


pdeplot(p, [], t, 'XYData', vu, 'Contour', 'off', 'Mesh', 'off');
colormap('jet');  % 设置颜色映射
colorbar;  % 显示颜色条
caxis([-1, 1]);
tmp = colorbar;   % 添加颜色条
% 设置颜色条刻度和刻度标签
tmp.Ticks = [-1,-0.5, 0, 0.5, 1];        % 设置刻度位置
% tmp.TickLabels = {'Low', 'Mid', 'High'};  % 设置刻度标签
title('$\Gamma_6^{(2)}$ GS', 'interpreter', 'latex', 'FontSize', 16);
xlabel('$x$',  'interpreter', 'latex');
ylabel('$y$',  'interpreter', 'latex');
view(2);  % 设置为三维视角
xlim([-1.04, 1.02]); % 设置 y 轴范围
ylim([-0.9, 0.9]); % 设置 y 轴范围
% axis equal;  % 坐标轴等比例显示


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 使用 print 导出 PDF
% 设置图形单位为厘米
set(gcf, 'Units', 'centimeters');
set(gcf, 'PaperUnits', 'centimeters');

% 设置图形窗口大小为 10cm x 7cm
set(gcf, 'Position', [0, 0, 28, 20]); % 图形窗口大小 (x, y, width, height)

% 设置导出的 PaperSize 和 PaperPosition
set(gcf, 'PaperSize', [30, 23]); % 页面大小
set(gcf, 'PaperPosition', [10, 10, 30, 20]); % 内容填满页面

% 导出 PDF
% print(gcf,'FigSolutionsTotal.pdf', '-dpdf', '-r0'); % 导出 PDF，去掉白边

fig = gcf;
set(fig, 'Units', 'centimeters');
fig_pos = get(fig, 'Position');   % [left, bottom, width, height]
set(fig, 'PaperUnits', 'centimeters');
set(fig, 'PaperPosition', [0 0 fig_pos(3) fig_pos(4)]);
set(fig, 'PaperSize', [fig_pos(3) fig_pos(4)]);
print(gcf, 'FigSolutionKGModel.png', '-dpng', '-r300');  


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function res = fn_int_exp(p, t, kx, ky)

res = 0;

f = @(x, y) exp(1j*kx*x + 1j*ky*y);

for it = 1:size(t, 2)
    
   ind1 = t(1, it);
   ind2 = t(2, it);
   ind3 = t(3, it);
   
   x1 = p(1, ind1);
   y1 = p(2, ind1);
   
   x2 = p(1, ind2);
   y2 = p(2, ind2);
   
   x3 = p(1, ind3);
   y3 = p(2, ind3);
   
   xQ1 = (4*x1 + x2 + x3)/6;
   yQ1 = (4*y1 + y2 + y3)/6;
   
   xQ2 = (x1 + 4*x2 + x3)/6;
   yQ2 = (y1 + 4*y2 + y3)/6;
   
   xQ3 = (x1 + x2 + 4*x3)/6;
   yQ3 = (y1 + y2 + 4*y3)/6;
   
   v1 = f(xQ1, yQ1);
   v2 = f(xQ2, yQ2);
   v3 = f(xQ3, yQ3);
   
   S = abs(fn_area(x1,y1,x2,y2,x3,y3));
   Value = (v1 + v2 + v3)/3;
   
   res = res + S*Value;
    
end

end


function res = fn_int_exp_phi(p, t, vu, kx, ky)

res = 0;

f = @(x, y) exp(1j*kx*x + 1j*ky*y);

for it = 1:size(t, 2)
    
   ind1 = t(1, it);
   ind2 = t(2, it);
   ind3 = t(3, it);
   
   x1 = p(1, ind1);
   y1 = p(2, ind1);
   
   x2 = p(1, ind2);
   y2 = p(2, ind2);
   
   x3 = p(1, ind3);
   y3 = p(2, ind3);
   
   z1 = vu(ind1);
   z2 = vu(ind2);
   z3 = vu(ind3);
   
   xQ1 = (4*x1 + x2 + x3)/6;
   yQ1 = (4*y1 + y2 + y3)/6;
   
   xQ2 = (x1 + 4*x2 + x3)/6;
   yQ2 = (y1 + 4*y2 + y3)/6;
   
   xQ3 = (x1 + x2 + 4*x3)/6;
   yQ3 = (y1 + y2 + 4*y3)/6;
   
   zQ1 = (4*z1 + z2 + z3)/6;
   zQ2 = (z1 + 4*z2 + z3)/6;
   zQ3 = (z1 + z2 + 4*z3)/6;
   
   v1 = f(xQ1, yQ1);
   v2 = f(xQ2, yQ2);
   v3 = f(xQ3, yQ3);
   
   S = abs(fn_area(x1,y1,x2,y2,x3,y3));
   Value = (v1*zQ1 + v2*zQ2 + v3*zQ3)/3;
   
   res = res + S*Value;
    
end

end


function [vkx, vky] = fn_generate_Kg(k0, th, N)

vth = [(2*pi/N*(0:(N-1)) + th), 2*pi/N*(0:-1:(1-N)) - th];
vkx = k0*cos(vth);
vky = k0*sin(vth);

end


function interpValue = interpolateFieldAtPoint(p, t, X, x0, y0)
% interpolateFieldAtPoint 插值平面上任意点的场值
% 
% 输入参数:
%   p  - 节点坐标矩阵 (2 x N)，每列为一个节点的 (x, y) 坐标
%   t  - 三角形顶点索引矩阵 (3 x M)，每列表示一个三角形的顶点索引
%   X  - 节点场值向量 (N x 1)，每个节点对应的场值
%   x0 - 查询点的 x 坐标
%   y0 - 查询点的 y 坐标
%
% 输出参数:
%   interpValue - 查询点 (x0, y0) 处插值得到的场值
%                 如果查询点不在网格范围内，则返回 NaN

    % 构建三角形网格
    triang = triangulation(t', p(1, :)', p(2, :)');
    
    % 查询点坐标
    queryPoint = [x0, y0];
    
    % 定位查询点所在的三角形
    tid = pointLocation(triang, queryPoint); % 返回三角形索引
    
    if isnan(tid)
        % 如果查询点不在网格范围内，返回 NaN
        interpValue = NaN;
        disp('查询点不在网格范围内');
    else
        % 获取所在三角形的顶点索引
        vertices = t(:, tid);
        
        % 提取三角形顶点的坐标和场值
        triCoords = [p(1, vertices)', p(2, vertices)']; % 三角形顶点坐标
        triValues = X(vertices); % 三角形顶点场值
        
        % 计算查询点的重心坐标
        baryCoords = cartesianToBarycentric(triang, tid, queryPoint);
        
        % 用重心坐标插值计算场值
        interpValue = baryCoords * triValues;
    end
end


function res = fn_Ext(p0, p, cSolution, cgMatrix)

cPsi = cSolution{2};
ca = cSolution{3};

res = 0;

for n = 1:length(cPsi)
    
    vu0 = cPsi{n};
    va = ca{n};
    
    vu = zeros(size(p,2), 1);
    for np0 = 1:size(p0,2)
        
        vxy = p0(:,np0);
        for ng = 1:length(va)
            
            tmp = cgMatrix{ng}*vxy;
            vDiss = sum((tmp - p).^2);
            PointFind = find(vDiss<1e-10,1);
            
            vu(PointFind) = vu(PointFind) + va(ng)*vu0(np0);
            
        end
        
    end
    
    res = res + vu;
    
end

end


function vu = fn_extension(p0, p, vu0, va, cgMatrix)

vu = zeros(size(p,2), 1);

for np0 = 1:size(p0,2)
    
   vxy = p0(:,np0);
   for ng = 1:length(va)
       
       tmp = cgMatrix{ng}*vxy;
       vDiss = sum((tmp - p).^2);
       PointFind = find(vDiss<1e-10,1);
       
       vu(PointFind) = vu(PointFind) + va(ng)*vu0(np0);
       
   end
    
end


end



function [p, t, pIndex] = fn_combine_pt(p1, t1, pIndex1, p2, t2, pIndex2)

N = size(p1, 2);
t2(1:3,:) = t2(1:3,:) + N;

p_store = zeros(2, size(p2, 2));
pIndex_store = zeros(1, size(p2, 2));
index_new = 0;

for point_p2 = 1:size(p2, 2)
    
   vxy = p2(:, point_p2);
   vDiss = (vxy - p1).^2;
   vDiss = sum(vDiss);
   
   IndexInP1 = find(vDiss<1e-11, 1);
   
   if isempty(IndexInP1)
       index_new = index_new + 1;
       p_store(:, index_new) = vxy;
       pIndex_store(:, index_new) = pIndex2(point_p2);
       t2(t2 == N + point_p2) = N + index_new;
   else
       t2(t2 == N + point_p2) = IndexInP1;
   end
    
end

p_store = p_store(:,1:index_new);
pIndex_store = pIndex_store(1:index_new);

p = [p1, p_store];
t = [t1, t2];
pIndex = [pIndex1, pIndex_store];

end


function [x0, y0] = fn_region0(x,y,N)

Angle = mod(angle(x + 1j*y), 2*pi);
Num = floor(Angle/(pi/N)) + 1;

if mod(Num, 2) == 1
   Angle0 = Angle - (Num-1)/2*2*pi/N; 
else
   Angle0 = -(Angle - Num/2*2*pi/N);
end

x0 = cos(Angle0)*sqrt(x^2+y^2);
y0 = sin(Angle0)*sqrt(x^2+y^2);

end



function res = fn_area(x1,y1,x2,y2,x3,y3)

res = 1/2*(x1*y2 + x2*y3 + x3*y1 - x1*y3 - x2*y1 - x3*y2);

end


function [m0, m1] = fn_zhuang(p, ~, t)

Num_t = size(t, 2);

vp1 = zeros(1,9*Num_t);
vp2 = zeros(1,9*Num_t);
vz0 = zeros(1,9*Num_t);
vz1 = zeros(1,9*Num_t);

ind_now = 0;

for it = 1:Num_t
    
    tv = t(:,it);
    p1 = tv(1);
    p2 = tv(2);
    p3 = tv(3);
    
    x1 = p(1, p1);
    y1 = p(2, p1);
    
    x2 = p(1, p2);
    y2 = p(2, p2);
    
    x3 = p(1, p3);
    y3 = p(2, p3);
    
    x12 = x1 - x2;
    x23 = x2 - x3;
    x31 = x3 - x1;
    
    y12 = y1 - y2;
    y23 = y2 - y3;
    y31 = y3 - y1;
    
    Delta = abs(fn_area(x1,y1,x2,y2,x3,y3));
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p1;
    vp2(ind_now) = p1;
    vz0(ind_now) = 1/6*Delta;
    vz1(ind_now) = (x23^2+y23^2)/4/Delta;
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p2;
    vp2(ind_now) = p2;
    vz0(ind_now) = 1/6*Delta;
    vz1(ind_now) = (x31^2+y31^2)/4/Delta; 
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p3;
    vp2(ind_now) = p3;
    vz0(ind_now) = 1/6*Delta;
    vz1(ind_now) =  (x12^2+y12^2)/4/Delta; 
       
    ind_now = ind_now + 1;
    vp1(ind_now) = p1;
    vp2(ind_now) = p2;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y23*y31 + x23*x31)/4/Delta; 
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p2;
    vp2(ind_now) = p1;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y23*y31 + x23*x31)/4/Delta; 
     
    ind_now = ind_now + 1;
    vp1(ind_now) = p2;
    vp2(ind_now) = p3;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y31*y12 + x31*x12)/4/Delta;
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p3;
    vp2(ind_now) = p2;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y31*y12 + x31*x12)/4/Delta;
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p3;
    vp2(ind_now) = p1;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y12*y23 + x12*x23)/4/Delta;
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p1;
    vp2(ind_now) = p3;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y12*y23 + x12*x23)/4/Delta;
     
end

m0 = sparse(vp1, vp2, vz0, size(p,2), size(p,2));
m1 = sparse(vp1, vp2, vz1, size(p,2), size(p,2));


end


function [rv,rm] = fn_cell_2_vec(tc,c_xv)

num_c = length(tc);
tot_element =0;
vec_length = 0;
for ic = 1:num_c
    tot_element = tot_element + length(tc{ic});
    vec_length = max(vec_length,size(c_xv{ic},1));
end

rv = zeros(tot_element,1);
rm = zeros(vec_length,tot_element);
point = 1;

for ic = 1:num_c
    [tv,t_ind] = sort(tc{ic});
    if ~isempty(tv)
        point2 = point + length(tv) - 1;
        rv(point:point2) = tv;
        tm = c_xv{ic};
        rm(:,point:point2) = tm(:,t_ind);
        point = point2 + 1;
    end
end

end


function res = fn_solve_3(eta, n1, n2, n3)

if n1 + n2 + n3>=0 || n1 + n2 + n3 + 3<=0
    error('wrong input n1 n2 n3');
end

c = 0;
cStep = 1.0;

x1 = fn_solve_single(eta, c, n1);
x2 = fn_solve_single(eta, c, n2);
x3 = fn_solve_single(eta, c, n3);

Direction = 0;

while abs(x1 + x2 + x3)>1e-10
   
    Direction_pre = Direction;
    if x1 + x2 + x3>0
        Direction = -1;
        c = c + Direction*cStep;
    else
        Direction = 1;
        c = c + Direction*cStep;
    end
    
    x1 = fn_solve_single(eta, c, n1);
    x2 = fn_solve_single(eta, c, n2);
    x3 = fn_solve_single(eta, c, n3);
    
    if Direction_pre*Direction == -1
        cStep = cStep/2;
    end
    
end

res = [x1, x2, x3];

end

function res = fn_solve_3_type2(eta, n1, n2, n3)

if n1 + n2 + n3>=3/2 || n1 + n2 + n3 + 3<=-3/2
    error('wrong input n1 n2 n3');
end

c = 0;
cStep = 1.0;

x1 = fn_solve_single(eta, c, n1);
x2 = fn_solve_single(eta, c, n2);
x3 = fn_solve_single(eta, c, n3);

Direction = 0;

while abs(x1 + x2 + x3 - 3*pi)>1e-10
   
    Direction_pre = Direction;
    if x1 + x2 + x3 - 3*pi>0
        Direction = -1;
        c = c + Direction*cStep;
    else
        Direction = 1;
        c = c + Direction*cStep;
    end
    
    x1 = fn_solve_single(eta, c, n1);
    x2 = fn_solve_single(eta, c, n2);
    x3 = fn_solve_single(eta, c, n3);
    
    if Direction_pre*Direction == -1
        cStep = cStep/2;
    end
    
end

res = [x1, x2, x3] - pi;

end


function res = fn_solve_single(eta, c, n)

% find the solution of the equation 
% eta*x - cot(x/2) == c in the region 2*pi*n<x<2*pi*(n+1)

y = fn_solve_single00(eta, c-2*pi*n*eta);
res = 2*pi*n + y;


end


function res = fn_solve_single00(eta, c)

f = @(x) eta*x - cot(x/2) - c;

x1 = 0.01;
while f(x1)>=0
    x1 = x1/2;
end

t = 0.01;
while f(2*pi - t)<=0
    t = t/2;
end
x2 = 2*pi - t;

while x2 - x1 > 1e-12
   
    x_mid = (x1 + x2)/2;
    if f(x_mid)>0
        x2 = x_mid;
    elseif f(x_mid)<0
        x1 = x_mid;
    else
        res = x_mid;
        return;
    end
    
end

y1 = f(x1);
y2 = f(x2);

res = (x1*y2 - y1*x2)/(y2-y1);

end



function res = fn_solve_single0(eta, c)

% find the solution of the equation 
% eta*x - cot(x/2) == c in the region 0<x<2*pi

f = @(y) eta*y - cot(y/2) - c;
fd = @(y) eta + 1/2/sin(y/2)^2;

x = pi;
F = f(x);

while abs(F)>1e-10
     
   dF = fd(x);
   dx = -F/dF;
   
   amp = 1.0;
   while 1
       x_try = x + amp*dx;
       if x_try>0 && x_try<2*pi
           x = x_try;
           break;
       end
       amp = amp/2;
   end
 
   F = f(x);
   disp(abs(F))
    
end

res = x;

end





function res = fn_lambda(k, c, k0, th, N)

tv = 0:(N-1);
res = c/2/1j/k0;
res = res./cos(2*pi/N*tv + th - pi/N*k - pi/2);

end


function res = fn_Scat(k, vlambda)

N = length(vlambda);
T = fn_T(N, k);
M = [eye(N), zeros(N, N); zeros(N, N), T];
res = [eye(N) + diag(vlambda), diag(vlambda); ...
    -diag(vlambda), eye(N) - diag(vlambda)];
res = M*res*transpose(M);

end


function res = fn_Psi(x, y, Ar, As, k0, th0)

N = length(Ar);
vr = 2*pi/N*(0:(N-1)) + th0;
vs = 2*pi/N*(0:(N-1)) - th0;

res = 0;
for n = 1:N
   k1 = k0*cos(vr(n));
   k2 = k0*sin(vr(n));
   res = res + Ar(n)*exp(1j*(x*k1 + y*k2));
end

for n = 1:N
   k1 = k0*cos(vs(n));
   k2 = k0*sin(vs(n));
   res = res + As(n)*exp(1j*(x*k1 + y*k2));
end

end


function res = fn_DerPsi(x, y, Ar, As, k0, th0, th_dir)

N = length(Ar);
vr = 2*pi/N*(0:(N-1)) + th0;
vs = 2*pi/N*(0:(N-1)) - th0;

res = 0;
for n = 1:N
   k1 = k0*cos(vr(n));
   k2 = k0*sin(vr(n));
   res = res + Ar(n)*exp(1j*(x*k1 + y*k2)) ...
       *1j*(cos(th_dir)*k1 + sin(th_dir)*k2);
end

for n = 1:N
   k1 = k0*cos(vs(n));
   k2 = k0*sin(vs(n));
   res = res + As(n)*exp(1j*(x*k1 + y*k2)) ...
       *1j*(cos(th_dir)*k1 + sin(th_dir)*k2);
end

end



function [Br, Bs] = fn_A2B(Ar, As, k, c, k0, th0)

N = length(Ar);

Br = zeros(1, N);
Bs = zeros(1, N);

for m = 0:(N-1)
    
   a = c/2/1j/k0/cos(2*pi/N*m + th0 - pi/N*k - pi/2);
   m2 = mod(k - m, N);
   Br(m+1) = (1 + a)*Ar(m+1) + a*As(m2 + 1);
    
end

for m = 0:(N-1)
    
   a = c/2/1j/k0/cos(2*pi/N*m - th0 - pi/N*k - pi/2);
   m2 = mod(k - m, N);
   Bs(m+1) = (1 + a)*As(m+1) + a*Ar(m2 + 1);
    
end

end







function res = fn_wave(x, y, Ar, As, vr, vs, k0)

N = length(Ar);

res = 0;
for n = 1:N
   k1 = k0*cos(vr(n));
   k2 = k0*sin(vr(n));
   res = res + Ar(n)*exp(1j*(x*k1 + y*k2));
end

for n = 1:N
   k1 = k0*cos(vs(n));
   k2 = k0*sin(vs(n));
   res = res + As(n)*exp(1j*(x*k1 + y*k2));
end

end


function [Ar, As] = fn_reflection(Ar0, As0, k)

N = length(Ar0);

Ar = zeros(1, N);
As = zeros(1, N);

for m = 0:(N-1)
    
    t = mod(k - m, N); 
    Ar(m + 1) = As0(t + 1);
    As(m + 1) = Ar0(t + 1);
    
end

end


function vtheta = fn_vtheta(c, k, k0, mangle)

[N, ~] = size(mangle);
N = N/2;

vlambda = [cos(pi/N*k + pi/2); sin(pi/N*k + pi/2)];

tmp = mangle*vlambda;
tmp = k0*tmp;

vtheta = (c/2/1j)./tmp;

end


function vbeta = fn_vbeta(k0, mangle, vd)

tmp = mangle*vd;
tmp = k0*tmp;

vbeta = exp(1j*tmp);

end


function vB = fn_scattering(vA, vtheta, k)

vB = zeros(size(vA));
N = length(vtheta);
N = N/2;

for n = 1:N
   tmp = vtheta(n);
   n2 = N + 1 + mod(k - n + 1, N);
   vB(n) = (1 + tmp)*vA(n) + tmp*vA(n2);
end

for n = (N+1):(2*N)
    tmp = vtheta(n);
    n2 = 1 + mod(k-n+N+1, N);
    vB(n) = (1+tmp)*vA(n) + tmp*vA(n2);
end

end


function vA = fn_tran(vA0, vbeta)

vA = vA0.*vbeta;

end


function vA = fn_transition(vA0, k0, mangle, vd)

vA = zeros(size(vA0));

for n = 1:N
   
    tmp = mangle(n, :)*vd;
    tmp = exp(1j*k0*tmp);
    vA(n) = vA0(n)*tmp;
    
end

end


function vA = fn_vA(vA0, k0, mangle, vx)

vA = zeros(size(vA0));

for n = 1:N
   
    tmp = mangle(n, :)*vx;
    tmp = exp(1j*k0*tmp);
    vA(n) = vA0(n)*tmp;
    
end

end


% function Psi = fn_Psi(vA, k0, mangle, vx)
% 
% Psi = 0;
% N = length(vA);
% 
% for n = 1:N
%     
%    tmp = mangle(n,:)*vx;
%    tmp = exp(1j*k0*tmp);
%    Psi = Psi + vA(n)*tmp;
%     
% end
% 
% end

function th = fn_s(m, th, N)

th = 2*pi/N*m - th;

end


function th = fn_r(m, th, N)

th = 2*pi/N*m + th;

end


function [vangle,mangle] = fn_g_angle(th0, N)

vangle = zeros(2*N, 1);

for m = 1:N
   vangle(m) = fn_r(m-1, th0, N); 
end

for m = (N+1):(2*N)
   vangle(m) = fn_s(m-N-1, th0, N); 
end

mangle = zeros(2*N, 2);

for m = 1:(2*N)
   
    mangle(m, :) = [cos(vangle(m)), sin(vangle(m))];
    
end

end



function  res = fn_D(N, n)

vx = 0:(N-1);
res = diag(exp(1j*2*pi/N*n*vx));

end

function res = fn_U(N)

vx = 0:(N-1);
vy = 0:(N-1);
mx = transpose(vx)*ones(1,N);
my = ones(N,1)*vy;

res = 1/sqrt(N)*exp(1j*2*pi/N*mx.*my);

end

function res = fn_T(N, n)

if n >= 0
    
    n = mod(n, N);
    tm = eye(N);
    res = tm(:,[(N-n+1):N, 1:(N-n)]);
    
else
    
    n = -n;
    n = mod(n, N);
    tm = eye(N);
    res = tm(:,[(N-n+1):N, 1:(N-n)]);
    res = transpose(res);
    
end

end


function map = addcolor(N)
% N为颜色种类

C = [0 0 0
0.549019607843137 0.549019607843137 0.549019607843137
0.627450980392157 0.627450980392157 0.627450980392157
0.756862745098039 0.756862745098039 0.756862745098039
0.827450980392157 0.827450980392157 0.827450980392157
0.882352941176471 0.882352941176471 0.882352941176471
0.905882352941177 0.905882352941177 0.905882352941177
0.972549019607843 0.972549019607843 0.972549019607843
1 0.980392156862745 0.980392156862745
0.737254901960784 0.560784313725490 0.560784313725490
0.941176470588235 0.501960784313726 0.501960784313726
0.803921568627451 0.360784313725490 0.360784313725490
0.647058823529412 0.164705882352941 0.164705882352941
0.698039215686275 0.133333333333333 0.133333333333333
0.501960784313726 0 0
0.545098039215686 0 0
1 0 0
1 0.894117647058824 0.882352941176471
0.980392156862745 0.501960784313726 0.447058823529412
1 0.388235294117647 0.278431372549020
0.913725490196078 0.588235294117647 0.478431372549020
1 0.498039215686275 0.313725490196078
1 0.270588235294118 0
1 0.627450980392157 0.478431372549020
0.627450980392157 0.321568627450980 0.176470588235294
1 0.960784313725490 0.933333333333333
0.823529411764706 0.411764705882353 0.117647058823529
0.545098039215686 0.270588235294118 0.0745098039215686
0.956862745098039 0.643137254901961 0.376470588235294
1 0.854901960784314 0.725490196078431
0.803921568627451 0.521568627450980 0.247058823529412
0.980392156862745 0.941176470588235 0.901960784313726
1 0.894117647058824 0.768627450980392
1 0.549019607843137 0
0.870588235294118 0.721568627450980 0.529411764705882
0.980392156862745 0.921568627450980 0.843137254901961
0.823529411764706 0.705882352941177 0.549019607843137
1 0.870588235294118 0.678431372549020
1 0.921568627450980 0.803921568627451
1 0.937254901960784 0.835294117647059
1 0.894117647058824 0.709803921568628
1 0.647058823529412 0
0.960784313725490 0.870588235294118 0.701960784313725
0.992156862745098 0.960784313725490 0.901960784313726
1 0.980392156862745 0.941176470588235
0.721568627450980 0.525490196078431 0.0431372549019608
0.854901960784314 0.647058823529412 0.125490196078431
1 0.972549019607843 0.862745098039216
1 0.843137254901961 0
1 0.980392156862745 0.803921568627451
0.941176470588235 0.901960784313726 0.549019607843137
0.933333333333333 0.909803921568627 0.666666666666667
0.741176470588235 0.717647058823529 0.419607843137255
1 1 0.941176470588235
0.960784313725490 0.960784313725490 0.862745098039216
1 1 0.878431372549020
0.980392156862745 0.980392156862745 0.823529411764706
0.501960784313726 0.501960784313726 0
0.749019607843137 0.749019607843137 0
1 1 0
0.419607843137255 0.556862745098039 0.137254901960784
0.603921568627451 0.803921568627451 0.196078431372549
0.333333333333333 0.419607843137255 0.184313725490196
0.678431372549020 1 0.184313725490196
0.498039215686275 1 0
0.486274509803922 0.988235294117647 0
0.941176470588235 1 0.941176470588235
0.560784313725490 0.737254901960784 0.560784313725490
0.596078431372549 0.984313725490196 0.596078431372549
0.564705882352941 0.933333333333333 0.564705882352941
0.133333333333333 0.545098039215686 0.133333333333333
0.196078431372549 0.803921568627451 0.196078431372549
0 0.392156862745098 0
0 0.501960784313726 0
0 1 0
0.180392156862745 0.545098039215686 0.341176470588235
0.235294117647059 0.701960784313725 0.443137254901961
0 1 0.498039215686275
0.960784313725490 1 0.980392156862745
0 0.980392156862745 0.603921568627451
0.400000000000000 0.803921568627451 0.666666666666667
0.498039215686275 1 0.831372549019608
0.250980392156863 0.878431372549020 0.815686274509804
0.125490196078431 0.698039215686275 0.666666666666667
0.282352941176471 0.819607843137255 0.800000000000000
0.941176470588235 1 1
0.878431372549020 1 1
0.686274509803922 0.933333333333333 0.933333333333333
0.184313725490196 0.309803921568627 0.309803921568627
0 0.501960784313726 0.501960784313726
0 0.545098039215686 0.545098039215686
0 0.749019607843137 0.749019607843137
0 1 1
0 0.807843137254902 0.819607843137255
0.372549019607843 0.619607843137255 0.627450980392157
0.690196078431373 0.878431372549020 0.901960784313726
0.678431372549020 0.847058823529412 0.901960784313726
0 0.749019607843137 1
0.529411764705882 0.807843137254902 0.921568627450980
0.529411764705882 0.807843137254902 0.980392156862745
0.274509803921569 0.509803921568627 0.705882352941177
0.941176470588235 0.972549019607843 1
0.117647058823529 0.564705882352941 1
0.466666666666667 0.533333333333333 0.600000000000000
0.439215686274510 0.501960784313726 0.564705882352941
0.690196078431373 0.768627450980392 0.870588235294118
0.392156862745098 0.584313725490196 0.929411764705882
0.254901960784314 0.411764705882353 0.882352941176471
0.972549019607843 0.972549019607843 1
0.901960784313726 0.901960784313726 0.980392156862745
0.0980392156862745    0.0980392156862745    0.439215686274510
0 0 0.501960784313726
0 0 0.545098039215686
0 0 0.803921568627451
0 0 1
0.415686274509804 0.352941176470588 0.803921568627451
0.282352941176471 0.239215686274510 0.545098039215686
0.482352941176471 0.407843137254902 0.933333333333333
0.576470588235294 0.439215686274510 0.858823529411765
0.400000000000000 0.200000000000000 0.600000000000000
0.541176470588235 0.168627450980392 0.886274509803922
0.294117647058824 0 0.509803921568627
0.600000000000000 0.196078431372549 0.800000000000000
0.580392156862745 0 0.827450980392157
0.729411764705882 0.333333333333333 0.827450980392157
0.847058823529412 0.749019607843137 0.847058823529412
0.866666666666667 0.627450980392157 0.866666666666667
0.933333333333333 0.509803921568627 0.933333333333333
0.501960784313726 0 0.501960784313726
0.545098039215686 0 0.545098039215686
0.749019607843137 0 0.749019607843137
1 0 1
0.854901960784314 0.439215686274510 0.839215686274510
0.780392156862745 0.0823529411764706    0.521568627450980
1 0.0784313725490196    0.576470588235294
1 0.411764705882353 0.705882352941177
1 0.941176470588235 0.960784313725490
0.858823529411765 0.439215686274510 0.576470588235294
0.862745098039216 0.0784313725490196    0.235294117647059
1 0.752941176470588 0.796078431372549
1 0.713725490196078 0.756862745098039
0.984313725490196 0.705882352941177 0.682352941176471
0.701960784313725 0.803921568627451 0.890196078431373
0.800000000000000 0.921568627450980 0.772549019607843
0.870588235294118 0.796078431372549 0.894117647058824
0.996078431372549 0.850980392156863 0.650980392156863
1 1 0.800000000000000
0.898039215686275 0.847058823529412 0.741176470588235
0.992156862745098 0.854901960784314 0.925490196078431
0.949019607843137 0.949019607843137 0.949019607843137
0.800000000000000 0.800000000000000 0.800000000000000
0.945098039215686 0.886274509803922 0.800000000000000
1 0.949019607843137 0.682352941176471
0.901960784313726 0.960784313725490 0.788235294117647
0.956862745098039 0.792156862745098 0.894117647058824
0.796078431372549 0.835294117647059 0.909803921568627
0.992156862745098 0.803921568627451 0.674509803921569
0.701960784313725 0.886274509803922 0.803921568627451
0.650980392156863 0.807843137254902 0.890196078431373
0.121568627450980 0.470588235294118 0.705882352941177
0.698039215686275 0.874509803921569 0.541176470588235
0.200000000000000 0.627450980392157 0.172549019607843
0.984313725490196 0.603921568627451 0.600000000000000
0.890196078431373 0.101960784313725 0.109803921568627
0.992156862745098 0.749019607843137 0.435294117647059
1 0.498039215686275 0
0.792156862745098 0.698039215686275 0.839215686274510
0.415686274509804 0.239215686274510 0.603921568627451
1 1 0.600000000000000
0.694117647058824 0.349019607843137 0.156862745098039
0.498039215686275 0.788235294117647 0.498039215686275
0.745098039215686 0.682352941176471 0.831372549019608
0.992156862745098 0.752941176470588 0.525490196078431
0.219607843137255 0.423529411764706 0.690196078431373
0.941176470588235 0.00784313725490196   0.498039215686275
0.749019607843137 0.356862745098039 0.0862745098039216
0.400000000000000 0.400000000000000 0.400000000000000
0.105882352941176 0.619607843137255 0.466666666666667
0.850980392156863 0.372549019607843 0.00784313725490196
0.458823529411765 0.439215686274510 0.701960784313725
0.905882352941177 0.160784313725490 0.541176470588235
0.400000000000000 0.650980392156863 0.117647058823529
0.901960784313726 0.670588235294118 0.00784313725490196
0.650980392156863 0.462745098039216 0.113725490196078
0.894117647058824 0.101960784313725 0.109803921568627
0.215686274509804 0.494117647058824 0.721568627450980
0.301960784313725 0.686274509803922 0.290196078431373
0.596078431372549 0.305882352941177 0.639215686274510
1 1 0.200000000000000
0.650980392156863 0.337254901960784 0.156862745098039
0.968627450980392 0.505882352941176 0.749019607843137
0.600000000000000 0.600000000000000 0.600000000000000
0.400000000000000 0.760784313725490 0.647058823529412
0.988235294117647 0.552941176470588 0.384313725490196
0.552941176470588 0.627450980392157 0.796078431372549
0.905882352941177 0.541176470588235 0.764705882352941
0.650980392156863 0.847058823529412 0.329411764705882
1 0.850980392156863 0.184313725490196
0.898039215686275 0.768627450980392 0.580392156862745
0.701960784313725 0.701960784313725 0.701960784313725
0.552941176470588 0.827450980392157 0.780392156862745
1 1 0.701960784313725
0.745098039215686 0.729411764705882 0.854901960784314
0.984313725490196 0.501960784313726 0.447058823529412
0.501960784313726 0.694117647058824 0.827450980392157
0.992156862745098 0.705882352941177 0.384313725490196
0.701960784313725 0.870588235294118 0.411764705882353
0.988235294117647 0.803921568627451 0.898039215686275
0.850980392156863 0.850980392156863 0.850980392156863
0.737254901960784 0.501960784313726 0.741176470588235
1 0.929411764705882 0.435294117647059
0.121568627450980 0.466666666666667 0.705882352941177
1 0.498039215686275 0.0549019607843137
0.172549019607843 0.627450980392157 0.172549019607843
0.839215686274510 0.152941176470588 0.156862745098039
0.580392156862745 0.403921568627451 0.741176470588235
0.549019607843137 0.337254901960784 0.294117647058824
0.890196078431373 0.466666666666667 0.760784313725490
0.498039215686275 0.498039215686275 0.498039215686275
0.737254901960784 0.741176470588235 0.133333333333333
0.0901960784313726    0.745098039215686 0.811764705882353
0.682352941176471 0.780392156862745 0.909803921568627
1 0.733333333333333 0.470588235294118
0.596078431372549 0.874509803921569 0.541176470588235
1 0.596078431372549 0.588235294117647
0.772549019607843 0.690196078431373 0.835294117647059
0.768627450980392 0.611764705882353 0.580392156862745
0.968627450980392 0.713725490196078 0.823529411764706
0.780392156862745 0.780392156862745 0.780392156862745
0.858823529411765 0.858823529411765 0.552941176470588
0.619607843137255 0.854901960784314 0.898039215686275
0.223529411764706 0.231372549019608 0.474509803921569
0.321568627450980 0.329411764705882 0.639215686274510
0.419607843137255 0.431372549019608 0.811764705882353
0.611764705882353 0.619607843137255 0.870588235294118
0.388235294117647 0.474509803921569 0.223529411764706
0.549019607843137 0.635294117647059 0.321568627450980
0.709803921568628 0.811764705882353 0.419607843137255
0.807843137254902 0.858823529411765 0.611764705882353
0.549019607843137 0.427450980392157 0.192156862745098
0.741176470588235 0.619607843137255 0.223529411764706
0.905882352941177 0.729411764705882 0.321568627450980
0.905882352941177 0.796078431372549 0.580392156862745
0.517647058823530 0.235294117647059 0.223529411764706
0.678431372549020 0.286274509803922 0.290196078431373
0.839215686274510 0.380392156862745 0.419607843137255
0.905882352941177 0.588235294117647 0.611764705882353
0.482352941176471 0.254901960784314 0.450980392156863
0.647058823529412 0.317647058823529 0.580392156862745
0.807843137254902 0.427450980392157 0.741176470588235
0.870588235294118 0.619607843137255 0.839215686274510
0.192156862745098 0.509803921568627 0.741176470588235
0.419607843137255 0.682352941176471 0.839215686274510
0.619607843137255 0.792156862745098 0.882352941176471
0.776470588235294 0.858823529411765 0.937254901960784
0.901960784313726 0.333333333333333 0.0509803921568627
0.992156862745098 0.552941176470588 0.235294117647059
0.992156862745098 0.682352941176471 0.419607843137255
0.992156862745098 0.815686274509804 0.635294117647059
0.192156862745098 0.639215686274510 0.329411764705882
0.454901960784314 0.768627450980392 0.462745098039216
0.631372549019608 0.850980392156863 0.607843137254902
0.780392156862745 0.913725490196078 0.752941176470588
0.458823529411765 0.419607843137255 0.694117647058824
0.619607843137255 0.603921568627451 0.784313725490196
0.737254901960784 0.741176470588235 0.862745098039216
0.854901960784314 0.854901960784314 0.921568627450980
0.388235294117647 0.388235294117647 0.388235294117647
0.588235294117647 0.588235294117647 0.588235294117647
0.741176470588235 0.741176470588235 0.741176470588235];

map = C(N,:);
end