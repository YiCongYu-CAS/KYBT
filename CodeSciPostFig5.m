set(0, 'DefaultAxesFontName', 'Times New Roman');
set(0, 'DefaultTextFontName', 'Times New Roman');
    
    
set(0, 'DefaultAxesFontSize', 20);     % 坐标轴刻度
set(0, 'DefaultTextFontSize', 20);     % 文字对象
set(0, 'DefaultAxesTitleFontSizeMultiplier', 1.25); % 标题与轴字号一致

set(0, 'DefaultAxesLabelFontSizeMultiplier', 1.25); % 标签字号=刻度字号 × 1.2



% 解N=6
N = 6;
Theta = pi/N;
L = 1;
c = 8;
eta = 2/c/L/sin(2*pi/N);

vx = fn_solve_3(eta, -2, -1, 1); % 解第一类BA方程

x1 = vx(1);
x2 = vx(2);
x3 = vx(3);
yh = sum(eta*vx - cot(vx/2))/3;

vx = linspace(1e-3, 2*pi-1e-3, 400);

f = @(x) eta*x - cot(x/2);

close(figure(1));
tf = figure(1); hold on;

set(tf, 'Position',  [300, 100, 900, 400]);  % 设置图形位置和大小
set(tf, 'Color', 'white');
set(gca, 'TickLabelInterpreter', 'latex');


Lx = 7*pi;
Ly = 4;

ax = findobj(tf, 'Type', 'axes');  % 找到 figure 里的 axes
set(ax, 'Box', 'on');
set(ax, 'YLim', [-Ly+0.5, Ly+0.2]);            % 设置 y 轴范围
set(ax, 'XLim', [-Lx, Lx]);  
set(ax, 'XTick', [-6*pi, -4*pi, -2*pi, 0, 2*pi, 4*pi, 6*pi]);
set(ax, 'XTickLabel', {'$-6\pi$', '$-4\pi$', '$-2\pi$', '0', '$2\pi$', '$4\pi$', '$6\pi$'});
set(get(ax, 'XLabel'), 'String', '$x$', 'Interpreter', 'latex');
set(get(ax, 'YLabel'), 'String', '$f(x)$', 'Interpreter', 'latex');

LW = 2;
co0 = addcolor(4);
plot(-3*[2*pi, 2*pi], [-Ly, Ly], '--', 'LineWidth', LW, 'Color', co0);
plot(-2*[2*pi, 2*pi], [-Ly, Ly], '--', 'LineWidth', LW, 'Color', co0);
plot(-1*[2*pi, 2*pi], [-Ly, Ly], '--', 'LineWidth', LW, 'Color', co0);
plot(0*[2*pi, 2*pi], [-Ly, Ly], '--', 'LineWidth', LW, 'Color', co0);
plot(1*[2*pi, 2*pi], [-Ly, Ly], '--', 'LineWidth', LW, 'Color', co0);
plot(2*[2*pi, 2*pi], [-Ly, Ly], '--', 'LineWidth', LW, 'Color', co0);
plot(3*[2*pi, 2*pi], [-Ly, Ly], '--', 'LineWidth', LW, 'Color', co0);

plot([-8*pi, 8*pi], [0, 0], '-black', 'LineWidth', 1.5);

LW = 2;
co1 = addcolor(17);

pl1 = plot(vx - 4*pi, f(vx - 4*pi), '-');
set(pl1, 'Color', co1, 'LineWidth', LW);

pl2 = plot(vx - 2*pi, f(vx - 2*pi), '-');
set(pl2, 'Color', co1, 'LineWidth', LW);

pl3 = plot(vx + 2*pi, f(vx + 2*pi), '-');
set(pl3, 'Color', co1, 'LineWidth', LW);


LW = 2;
tp = plot(vx -6*pi, f(vx - 6*pi), '-');
set(tp, 'Color', co0, 'LineWidth', LW);

tp = plot(vx -8*pi, f(vx - 8*pi), '-');
set(tp, 'Color', co0, 'LineWidth', LW);

tp = plot(vx + 0*pi, f(vx + 0*pi), '-');
set(tp, 'Color', co0, 'LineWidth', LW);
tp = plot(vx + 4*pi, f(vx + 4*pi), '-');
set(tp, 'Color', co0, 'LineWidth', LW);
tp = plot(vx + 6*pi, f(vx + 6*pi), '-');
set(tp, 'Color', co0, 'LineWidth', LW);

co2 = addcolor(114);
co2in = addcolor(107);
Area = 36;

pl2 = plot([-Lx, Lx], [yh, yh], '-');
set(pl2, 'Color', co2in, 'LineWidth', 3);

tp = plot(x1*ones(1,2), [0, yh], '--');
set(tp, 'Color', 'black', 'LineWidth', 2);
tp = plot(x2*ones(1,2), [0, yh], '--');
set(tp, 'Color', 'black', 'LineWidth', 2);
tp = plot(x3*ones(1,2), [0, yh], '--');
set(tp, 'Color', 'black', 'LineWidth', 2);

scatter(x1, yh, Area, ...            % 200是点的面积, 可调大些
    'MarkerEdgeColor', co2, ... % 边界色
    'MarkerFaceColor', co2in, ... % 填充色（红色，可自选）
    'LineWidth', 2);

scatter(x2, yh, Area, ...            % 200是点的面积, 可调大些
    'MarkerEdgeColor', co2, ... % 边界色
    'MarkerFaceColor', co2in, ... % 填充色（红色，可自选）
    'LineWidth', 2);

scatter(x3, yh, Area, ...            % 200是点的面积, 可调大些
    'MarkerEdgeColor', co2, ... % 边界色
    'MarkerFaceColor', co2in, ... % 填充色（红色，可自选）
    'LineWidth', 2);

scatter(-7*pi, yh, 10, ...            % 200是点的面积, 可调大些
    'MarkerEdgeColor', 'black', ... % 边界色
    'MarkerFaceColor', 'black', ... % 填充色（红色，可自选）
    'LineWidth', 2);


xshift = 0.7;
yshift = -0.75;
text(x1 + xshift, 0 + yshift, '$x_1$', ...
    'Interpreter', 'latex', ...
    'FontSize', 20, ...
    'Color', 'black', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom');

text(x2 + xshift, 0 + yshift, '$x_2$', ...
    'Interpreter', 'latex', ...
    'FontSize', 20, ...
    'Color', 'black', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom');

text(x3 + xshift, 0 + yshift, '$x_3$', ...
    'Interpreter', 'latex', ...
    'FontSize', 20, ...
    'Color', 'black', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom');

text(-7*pi + xshift, yh + yshift, '$w$', ...
    'Interpreter', 'latex', ...
    'FontSize', 20, ...
    'Color', 'black', ...
    'HorizontalAlignment', 'center', ...
    'VerticalAlignment', 'bottom');


lgd = legend([pl1, pl2], {'$f(x)$', '$w$'});
set(lgd, 'Position', [0.795 0.79 0.1 0.1],  'Interpreter', 'latex'); % [左 下 宽 高]

fig = gcf;
set(fig, 'Units', 'centimeters');
fig_pos = get(fig, 'Position');   % [left, bottom, width, height]
set(fig, 'PaperUnits', 'centimeters');
set(fig, 'PaperPosition', [0 0 fig_pos(3) fig_pos(4)]);
set(fig, 'PaperSize', [fig_pos(3) fig_pos(4)]);
print(gcf, 'FigLQZCBAE.pdf', '-dpdf');   

% k0 = sqrt(2/3*sum(vx.^2))/L;
% th0 = pi/N + acos(vx(1)/k0/L);
% 
% vbeta = zeros(1, N);
% vthi = zeros(1, N);
% for n = 0:(N-1)
%     vbeta(n+1) = exp(-1j*L/2*k0*cos(2*pi/N*n + th0 + Theta));
%     vthi(n+1) = 2*1j*k0/c*sin(2*pi/N*n + th0 - Theta);
% end
% 
% % 验证第一类BA方程
% for m = 0:(N-1)
%    LHS = vthi(m+1);
%    m2 = mod(m - 2, N);
%    RHS = 1/(vbeta(m2+1)^2-1) - 1/(vbeta(m+1)^2-1);
%    fprintf('check BA type-I m = %d, error = %e \n', m, abs(LHS - RHS));
% end
% 
% % 解第二类BA方程
% vx2 = fn_solve_3_type2(eta, -1, 0, 2);
% k0 = sqrt(2/3*sum(vx2.^2))/L;
% th0 = pi/N + acos(vx2(1)/k0/L);
% 
% vbeta = zeros(1, N);
% vthi = zeros(1, N);
% for n = 0:(N-1)
%     vbeta(n+1) = exp(-1j*L/2*k0*cos(2*pi/N*n + th0 + Theta));
%     vthi(n+1) = 2*1j*k0/c*sin(2*pi/N*n + th0 - Theta);
% end
% 
% % 验证第二类BA方程
% for m = 0:(N-1)
%    LHS = vthi(m+1);
%    m2 = mod(m - 2, N);
%    RHS = 1/(vbeta(m+1)^2+1) - 1/(vbeta(m2+1)^2+1);
%    fprintf('check BA type-II m = %d, error = %e \n', m, abs(LHS - RHS));
% end
% 
% % 还原所有的Ar和As
% omega0 = cos(L/2*k0*cos(2*pi/N*0 + th0 + Theta));
% omega1 = cos(L/2*k0*cos(2*pi/N*1 + th0 + Theta));
% omega2 = cos(L/2*k0*cos(2*pi/N*2 + th0 + Theta));
% omega3 = cos(L/2*k0*cos(2*pi/N*3 + th0 + Theta));
% omega4 = cos(L/2*k0*cos(2*pi/N*4 + th0 + Theta));
% omega5 = cos(L/2*k0*cos(2*pi/N*5 + th0 + Theta));
% 
% A0 = 1;
% A1 = omega5/omega1*A0;
% A2 = omega0/omega2*A1;
% 
% Ar = zeros(1, N);
% As = zeros(1, N);
% Ar(1) = A0;
% Ar(2) = A1;
% Ar(3) = A2;
% Ar(4) = A0;
% Ar(5) = A1;
% Ar(6) = A2;
% 
% As(1) = -A0;
% As(2) = -A2;
% As(3) = -A1;
% As(4) = -A0;
% As(5) = -A2;
% As(6) = -A1;
% 
% tv = exp(-1j*L*cos(Theta)*k0*cos(2*pi/N*(0:(N-1)) + th0));
% Ar = Ar.*tv;
% tv = exp(-1j*L*cos(Theta)*k0*cos(2*pi/N*(0:(N-1)) - th0));
% As = As.*tv;
% 
% % 回到原点，求两个区域的散射Ar, As
% [Ar2, As2] = fn_A2B(Ar, As, 1, c, k0, th0);
% 
% % 画图
% Lx = 1;
% Ly = 1;
% 
% vx = linspace(-Lx, Lx, 400);
% vy = linspace(-Ly, Ly, 400);
% [mx, my] = meshgrid(vx, vy);
% 
% mr = zeros(length(vy), length(vx));
% 
% for ix = 1:length(vx)
%     for iy = 1:length(vy)
%         
%         x = mx(iy, ix);
%         y = my(iy, ix);
%         tmp = y - tan(pi/N)*x;
%         if tmp<=0
%             mr(iy, ix) = fn_Psi(x, y, Ar, As, k0, th0);
%         else
%             mr(iy, ix) = fn_Psi(x, y, Ar2, As2, k0, th0);
%         end
%         
%     end
% end
% 
% figure(2);
% h1 = pcolor(mx, my, real(mr));
% set(h1, 'linestyle', 'none');
% 
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% %%%%%% 求解BA方程得到能谱
% 
% vp = 1:10;
% vq = 1:10;
% 
% num_eig = 0;
% vr = zeros(1, 1000);
% vr_type = zeros(1, 1000);
% 
% for ip = 1:length(vp)
%     for iq = 1:length(vq)
%         
%         p = vp(ip);
%         q = vq(iq);
%         
%         %  偶宇称解
%         if mod(p-q, 3)~=0 && p<q
%            n2 = floor((p-q)/3);
%            n1 = n2 - p;
%            n3 = n2 + q;
%            tv = fn_solve_3(eta, n1, n2, n3);
%            k0 = sqrt(2/3*sum(tv.^2))/L;
%            num_eig = num_eig + 1;
%            vr(num_eig) = k0^2;
%            vr_type(num_eig) = 1;
%            
%         end
%            
%         % 奇宇称解
%         if p<q
%             
%             n2 = floor((p-q)/3 + 1/2);
%             n1 = n2 - p;
%             n3 = n2 + q;
%             
%             tv = fn_solve_3_type2(eta, n1, n2, n3);
%             k0 = sqrt(2/3*sum(tv.^2))/L;
%             num_eig = num_eig + 1;
%             vr(num_eig) = k0^2;
%             vr_type(num_eig) = -1;
%         
%         end
%         
%         % 铺砖块解
%         if p<q && mod(q-p, 3) == 0
%               
%             n2 = (p - q)/3;
%             n1 = n2 - p;
%             n3 = n2 + q;
%             
%             k0 = sqrt(2/3*(2*pi)^2*(n1^2+n2^2+n3^2));
%             num_eig = num_eig + 1;
%             vr(num_eig) = k0^2;
%             vr_type(num_eig) = 0;
%             
%         end         
%         
%     end
% end
% 
% vr = vr(1:num_eig);
% [vr, tv] = sort(vr);
% vr_type = vr_type(tv);
% % vr 是能谱的解析解
% % vr_type 是波函数的类型， 1： 偶， -1： 奇， 0： 砖块
% 
% 
% Theta = pi/3;
% triangle1 = [2;3;0;cos(Theta);cos(Theta);0;0;sin(Theta)];
% triangle2 = [2;3;0;cos(Theta);0;0;sin(Theta);sin(Theta)];
% 
% gd = [triangle1, triangle2];
% ns = char('triangle1', 'triangle2');
% ns = ns';
% 
% sf = 'triangle1 + triangle2';
% 
% [dl,bt] = decsg(gd,sf,ns);
% 
% [p,e,t] = initmesh(dl,"Hmax",0.005);
% 
% vedge = zeros(1, size(p, 2));
% vedge2 = zeros(1, size(p, 2));
% 
% for ip = 1:size(p,2)
%     
%    x = p(1, ip);
%    y = p(2, ip);
%    
%    if abs(x)<1e-10 || abs(y)<1e-10 ...
%            || abs(x - cos(Theta))<1e-10 || abs(y - sin(Theta))<1e-10 
%        vedge(ip) = 1;
%    end
%    
%    if abs(cos(Theta)*y - sin(Theta)*x)<1e-10
%        vedge2(ip) = 1;
%    end
%     
% end
% 
% tv = 1:size(p, 2);
% vedge_ind = tv(vedge == 1);
% vedge2_ind = tv(vedge2 == 1);
% 
% vinternal = tv(vedge == 0);
% 
% vei_ind = find(e(5,:) == 5);
% tve = e(:, vei_ind);
% [~, tsort] = sort(tve(3, :));
% tve = tve(:,tsort);
% 
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% vp1 = zeros(1, length(tve)*4);
% vp2 = zeros(1, length(tve)*4);
% vz = zeros(1, length(tve)*4);
% 
% num_insert = 0;
% 
% mDelta = zeros(size(p,2), size(p, 2));
% for ie = 1:length(tve)
%    
%     p1 = tve(1, ie);
%     p2 = tve(2, ie);
%     
%     x1 = p(1, p1);
%     y1 = p(2, p1);
%     
%     x2 = p(1, p2);
%     y2 = p(2, p2);
%     
%     d = sqrt((x1-x2)^2 + (y1-y2)^2);
% 
%     mDelta(p1, p1) = mDelta(p1, p1) + 1/3*d;
%     num_insert = num_insert + 1;
%     vp1(num_insert) = p1;
%     vp2(num_insert) = p1;
%     vz(num_insert) = 1/3*d;
%     
%     mDelta(p2, p2) = mDelta(p2, p2) + 1/3*d;
%     num_insert = num_insert + 1;
%     vp1(num_insert) = p2;
%     vp2(num_insert) = p2;
%     vz(num_insert) = 1/3*d;
%     
%     mDelta(p1, p2) = mDelta(p1, p2) + 1/6*d;
%     num_insert = num_insert + 1;
%     vp1(num_insert) = p1;
%     vp2(num_insert) = p2;
%     vz(num_insert) = 1/6*d;
%     
%     mDelta(p2, p1) = mDelta(p2, p1) + 1/6*d;
%     num_insert = num_insert + 1;
%     vp1(num_insert) = p2;
%     vp2(num_insert) = p1;
%     vz(num_insert) = 1/6*d;
%     
% end
% 
% mDelta = sparse(vp1, vp2, vz, size(p,2), size(p,2));
% 
% [m0, m1] = fn_zhuang(p, e, t);
% 
% M0 = m0(vinternal, vinternal);
% M1 = m1(vinternal, vinternal);
% MDelta = mDelta(vinternal, vinternal);
% 
% A = M1 + c*MDelta;
% B = M0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('\n ============================ \n');
% fprintf('begin to search eigs\n');
% 
% search_step = 50;
% search_eig_min = 20; % initial
% region_begin = search_eig_min;
% 
% num_step = 10;
% c_lmb = cell(1,num_step);
% c_xv = cell(1,num_step);
% 
% for istep = 1:num_step
%      
%     while 1
%         search_eig_max = search_eig_min + search_step;
%         [xv,lmb,iresult]...
%             = sptarn(A,B,search_eig_min,search_eig_max,1,100*eps,100);
% 
%         if iresult>=0
%             break;
%         else
%             search_step = search_step/2;
%         end
% 
%     end
%     
%     c_lmb{istep} = lmb;
%     c_xv{istep} = xv;
%     fprintf('=======================================\n');
%     fprintf('finished interval [%4.2f, %4.2f], find eigs %d \n',...
%         search_eig_min,search_eig_max,iresult);
%   
%     search_eig_min = search_eig_max;
% end
% 
% [rv,rm] = fn_cell_2_vec(c_lmb,c_xv);
% rv = rv';
% 
% rv_ana = vr(1:length(rv));
% rv_ana_type = vr_type(1:length(rv));
% 
% close(figure(3));
% figure(3); hold on;
% plot(1:length(rv), rv, '-r');
% for iv = 1:length(rv)
%     
%    if rv_ana_type(iv) == 1
%        plot(iv, rv_ana(iv), 'bo');
%    elseif rv_ana_type(iv) == -1
%        plot(iv, rv_ana(iv), 'go');
%    else
%        plot(iv, rv_ana(iv), 'y*');
%    end
%     
% end
% 
% title('comparison of analytical and numerical spectrum');
% xlabel('index');
% ylabel('spectrum');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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
