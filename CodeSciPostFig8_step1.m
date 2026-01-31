%%%%%%  Part 1: generate grid  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Theta = pi/6;
triangle1 = [2;3;0;1;cos(Theta)^2;0;0;cos(Theta)*sin(Theta)];

c = 1;

gd = triangle1;
ns = char('triangle1');
ns = ns';

sf = 'triangle1';

[dl,bt] = decsg(gd,sf,ns);

[p,e,t] = initmesh(dl,"Hmax",0.05);

% close(figure(4));
% figure(4);
% pdeplot(p, [], t);
% axis equal;
% title('Mesh of a Triangle');

% hold on;
% numPoints = size(p, 2);
% for i = 1:numPoints
%     text(p(1, i), p(2, i), num2str(i), 'VerticalAlignment', 'bottom', ...
%         'HorizontalAlignment', 'right', 'FontSize', 16, 'Color', 'red');
% end
% 
% numEdges = size(e, 2);
% for i = 1:numEdges
%     % 计算边的中点坐标
%     midX = (p(1, e(1, i)) + p(1, e(2, i))) / 2;
%     midY = (p(2, e(1, i)) + p(2, e(2, i))) / 2;
%     % 显示边的标号
%     text(midX, midY, num2str(i), 'FontSize', 16, 'Color', 'blue');
% end
% 
% numTriangles = size(t, 2);
% for i = 1:numTriangles
%     % 计算三角形的重心坐标
%     centroidX = mean(p(1, t(1:3, i)));
%     centroidY = mean(p(2, t(1:3, i)));
%     % 显示三角形的标号
%     text(centroidX, centroidY, num2str(i), 'FontSize', 16, 'Color', 'black');
% end

%%%%%% END Part 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Part 2: prepare datas %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

vedge = zeros(1, size(p,2));
vedge0 = zeros(1, size(p, 2));
vedge1 = zeros(1, size(p, 2));

for ip = 1:size(p,2)
    
   x = p(1, ip);
   y = p(2, ip);
   
   if abs(y)<1e-10
       vedge1(ip) = 1;
   end
   
   if abs(cos(Theta)*y - sin(Theta)*x)<1e-10
       vedge0(ip) = 1;
   end
   
   if abs(y*cos(Theta + pi/2) - (x-1)*sin(Theta + pi/2))<1e-10
       vedge(ip) = 1;
   end
    
end

tv = 1:size(p, 2);
vedge0_ind = tv(vedge0 == 1);
vedge1_ind = tv(vedge1 == 1);
vinternal = tv(vedge == 0);
vBoundary = tv(vedge == 1);
vboundary = union(vedge0_ind, vedge1_ind);

vtop = intersect(vedge1_ind, vedge0_ind);
if length(vtop)~=1
    error('only 1 top is permitted');
end

vBigboundary = vboundary;
vboundary = setdiff(vboundary, vtop);
vboundary0 = setdiff(vedge0_ind, vtop);
vboundary1 = setdiff(vedge1_ind, vtop);

vPointType = 2*ones(1,size(p,2));
vPointType(vboundary0) = 0;
vPointType(vboundary1) = 1;
vPointType(vtop) = -1;

tm = p(:, vedge1_ind);
tvx = tm(1,:);
[~, vsort] = sort(tvx);
vedge1_ind = vedge1_ind(vsort);

vTop = vtop;
vEdge0 = setdiff(vboundary0, vBoundary);
vEdge1 = setdiff(vboundary1, vBoundary);
vInternal = 1:size(p, 2);
vInternal = setdiff(vInternal, vBoundary);
vInternal = setdiff(vInternal, vTop);
vInternal = setdiff(vInternal, vEdge0);
vInternal = setdiff(vInternal, vEdge1);

%%%%%% END Part 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Part 3: generate matrix <v_m|v_n>, <v_m|T|v_n>, <v_m|V|v_n> %%%%%%%

Num_t = size(t, 2);

vp1 = zeros(1,9*Num_t);
vp2 = zeros(1,9*Num_t);
vz0 = zeros(1,9*Num_t);

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
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p2;
    vp2(ind_now) = p2;
    vz0(ind_now) = 1/6*Delta;
   
    ind_now = ind_now + 1;
    vp1(ind_now) = p3;
    vp2(ind_now) = p3;
    vz0(ind_now) = 1/6*Delta;
       
    ind_now = ind_now + 1;
    vp1(ind_now) = p1;
    vp2(ind_now) = p2;
    vz0(ind_now) = 1/12*Delta;
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p2;
    vp2(ind_now) = p1;
    vz0(ind_now) = 1/12*Delta;
     
    ind_now = ind_now + 1;
    vp1(ind_now) = p2;
    vp2(ind_now) = p3;
    vz0(ind_now) = 1/12*Delta;
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p3;
    vp2(ind_now) = p2;
    vz0(ind_now) = 1/12*Delta;
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p3;
    vp2(ind_now) = p1;
    vz0(ind_now) = 1/12*Delta;
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p1;
    vp2(ind_now) = p3;
    vz0(ind_now) = 1/12*Delta;
     
end

m00 = sparse(vp1, vp2, vz0, size(p,2), size(p,2));


Num = length(vedge1_ind) - 1;
vp1 = zeros(1, Num*4);
vp2 = zeros(1, Num*4);
vz = zeros(1, Num*4);

num_insert = 0;
for ie = 1:Num
   
%     p1 = tve(1, ie);
%     p2 = tve(2, ie);

    p1 = vedge1_ind(ie);
    p2 = vedge1_ind(ie+1);
    
    x1 = p(1, p1);
    y1 = p(2, p1);
    
    x2 = p(1, p2);
    y2 = p(2, p2);
    
    d = sqrt((x1-x2)^2 + (y1-y2)^2);

    num_insert = num_insert + 1;
    vp1(num_insert) = p1;
    vp2(num_insert) = p1;
    vz(num_insert) = 1/3*d;  
    if p1 == vtop
        vz(num_insert) = vz(num_insert)*6;
    end
    
    num_insert = num_insert + 1;
    vp1(num_insert) = p2;
    vp2(num_insert) = p2;
    vz(num_insert) = 1/3*d;
    if p2 == vtop
        vz(num_insert) = vz(num_insert)*6;
    end
    
    num_insert = num_insert + 1;
    vp1(num_insert) = p1;
    vp2(num_insert) = p2;
    vz(num_insert) = 1/6*d;
    
    num_insert = num_insert + 1;
    vp1(num_insert) = p2;
    vp2(num_insert) = p1;
    vz(num_insert) = 1/6*d;
    
end

mDelta = sparse(vp1, vp2, vz, size(p,2), size(p,2));

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
    if vPointType(p1) == 0 || vPointType(p1) == 1
        vz0(ind_now) = 2*vz0(ind_now);
        vz1(ind_now) = 2*vz1(ind_now);
    end
    if vPointType(p1) == -1
        vz0(ind_now) = 12*vz0(ind_now);
        vz1(ind_now) = 12*vz1(ind_now);
    end
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p2;
    vp2(ind_now) = p2;
    vz0(ind_now) = 1/6*Delta;
    vz1(ind_now) = (x31^2+y31^2)/4/Delta; 
    if vPointType(p2) == 0 || vPointType(p2) == 1
        vz0(ind_now) = 2*vz0(ind_now);
        vz1(ind_now) = 2*vz1(ind_now);
    end
    if vPointType(p2) == -1
        vz0(ind_now) = 12*vz0(ind_now);
        vz1(ind_now) = 12*vz1(ind_now);
    end
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p3;
    vp2(ind_now) = p3;
    vz0(ind_now) = 1/6*Delta;
    vz1(ind_now) =  (x12^2+y12^2)/4/Delta; 
    if vPointType(p3) == 0 || vPointType(p3) == 1
        vz0(ind_now) = 2*vz0(ind_now);
        vz1(ind_now) = 2*vz1(ind_now);
    end
    if vPointType(p3) == -1
        vz0(ind_now) = 12*vz0(ind_now);
        vz1(ind_now) = 12*vz1(ind_now);
    end
       
    ind_now = ind_now + 1;
    vp1(ind_now) = p1;
    vp2(ind_now) = p2;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y23*y31 + x23*x31)/4/Delta; 
    if (vPointType(p1) == 0 && vPointType(p2) == 0) ...
        || (vPointType(p1) == 1 && vPointType(p2) == 1) ...
        || (vPointType(p1) == -1 && vPointType(p2) == 0) ...
        || (vPointType(p1) == -1 && vPointType(p2) == 1) ...
        || (vPointType(p1) == 0 && vPointType(p2) == -1) ...
        || (vPointType(p1) == 1 && vPointType(p2) == -1)
        vz0(ind_now) = 2*vz0(ind_now);
        vz1(ind_now) = 2*vz1(ind_now);
    end
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p2;
    vp2(ind_now) = p1;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y23*y31 + x23*x31)/4/Delta; 
    if (vPointType(p2) == 0 && vPointType(p1) == 0) ...
        || (vPointType(p2) == 1 && vPointType(p1) == 1) ...
        || (vPointType(p2) == -1 && vPointType(p1) == 0) ...
        || (vPointType(p2) == -1 && vPointType(p1) == 1) ...
        || (vPointType(p2) == 0 && vPointType(p1) == -1) ...
        || (vPointType(p2) == 1 && vPointType(p1) == -1)
        vz0(ind_now) = 2*vz0(ind_now);
        vz1(ind_now) = 2*vz1(ind_now);
    end
     
    ind_now = ind_now + 1;
    vp1(ind_now) = p2;
    vp2(ind_now) = p3;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y31*y12 + x31*x12)/4/Delta;
    if (vPointType(p2) == 0 && vPointType(p3) == 0) ...
        || (vPointType(p2) == 1 && vPointType(p3) == 1) ...
        || (vPointType(p2) == -1 && vPointType(p3) == 0) ...
        || (vPointType(p2) == -1 && vPointType(p3) == 1) ...
        || (vPointType(p2) == 0 && vPointType(p3) == -1) ...
        || (vPointType(p2) == 1 && vPointType(p3) == -1)
        vz0(ind_now) = 2*vz0(ind_now);
        vz1(ind_now) = 2*vz1(ind_now);
    end
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p3;
    vp2(ind_now) = p2;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y31*y12 + x31*x12)/4/Delta;
    if (vPointType(p3) == 0 && vPointType(p2) == 0) ...
        || (vPointType(p3) == 1 && vPointType(p2) == 1) ...
        || (vPointType(p3) == -1 && vPointType(p2) == 0) ...
        || (vPointType(p3) == -1 && vPointType(p2) == 1) ...
        || (vPointType(p3) == 0 && vPointType(p2) == -1) ...
        || (vPointType(p3) == 1 && vPointType(p2) == -1)
        vz0(ind_now) = 2*vz0(ind_now);
        vz1(ind_now) = 2*vz1(ind_now);
    end
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p3;
    vp2(ind_now) = p1;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y12*y23 + x12*x23)/4/Delta;
    if (vPointType(p3) == 0 && vPointType(p1) == 0) ...
        || (vPointType(p3) == 1 && vPointType(p1) == 1) ...
        || (vPointType(p3) == -1 && vPointType(p1) == 0) ...
        || (vPointType(p3) == -1 && vPointType(p1) == 1) ...
        || (vPointType(p3) == 0 && vPointType(p1) == -1) ...
        || (vPointType(p3) == 1 && vPointType(p1) == -1)
        vz0(ind_now) = 2*vz0(ind_now);
        vz1(ind_now) = 2*vz1(ind_now);
    end
    
    ind_now = ind_now + 1;
    vp1(ind_now) = p1;
    vp2(ind_now) = p3;
    vz0(ind_now) = 1/12*Delta;
    vz1(ind_now) =  (y12*y23 + x12*x23)/4/Delta;
    if (vPointType(p1) == 0 && vPointType(p3) == 0) ...
        || (vPointType(p1) == 1 && vPointType(p3) == 1) ...
        || (vPointType(p1) == -1 && vPointType(p3) == 0) ...
        || (vPointType(p1) == -1 && vPointType(p3) == 1) ...
        || (vPointType(p1) == 0 && vPointType(p3) == -1) ...
        || (vPointType(p1) == 1 && vPointType(p3) == -1)
        vz0(ind_now) = 2*vz0(ind_now);
        vz1(ind_now) = 2*vz1(ind_now);
    end
     
end

m0 = sparse(vp1, vp2, vz0, size(p,2), size(p,2));
m1 = sparse(vp1, vp2, vz1, size(p,2), size(p,2));

%%%%%% END Part 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% Part 4: prepare datas %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 以上完成了初始区域的基的交叠积分的计算
% 接下来完成整个区域的基的交叠积分的计算
% 先做准备，生成所有群元素操作对应的onehot编码

N = 6;
T = diag(ones(1,N-1), 1);
T(N, 1) = 1;
Tm = T';
I0 = zeros(N, N);

R = [Tm, I0; I0, T];
S = [I0, eye(N); eye(N), I0];

mOneHotEdge0 = zeros(2*N, 2*N);
mOneHotEdge1 = zeros(2*N, 2*N);
mOneHotInt = zeros(2*N, 2*N);
mOneHotTop = zeros(2*N, 2*N);

vOneHotZero0 = zeros(2*N, 1);
vOneHotZero1 = zeros(2*N, 1);

vOneHotZero0(1) = 1;
vOneHotZero0(12) = 1;

vOneHotZero1(1) = 1;
vOneHotZero1(7) = 1;

for k = 1:N
    mGroup = R^(k-1);
    mOneHotEdge0(k,:) = mGroup*vOneHotZero0;
    mOneHotEdge1(k,:) = mGroup*vOneHotZero1;
    tmp = zeros(2*N, 1);
    tmp(k) = 1;
    mOneHotInt(k,:) = tmp;
    mOneHotTop(k,:) = ones(2*N, 1);
end

for k = 1:N
   mGroup = S*R^(k-1);
   mOneHotEdge0(k+N,:) = mGroup*vOneHotZero0;
   mOneHotEdge1(k+N,:) = mGroup*vOneHotZero1;
   tmp = zeros(2*N, 1);
   tmp(k+N) = 1;
   mOneHotInt(k+N,:) = tmp;
   mOneHotTop(k+N,:) = ones(2*N, 1);
end

%%%%%% END Part 4 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%% PART 5: 1D representation %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 接下来分别计算每一个不可约表示空间中的解

%%%%%  1维表示， 第1类
va = [1,1,1,1,1,1,1,1,1,1,1,1];
va1 = va;
va2 = va;

%  calculate M0
[row0, col0] = find(m0);
vp1 = zeros(1, length(row0));
vp2 = zeros(1, length(row0));
vz0 = zeros(1, length(row0));

for n = 1:length(row0)
    
    p1 = row0(n);
    p2 = col0(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0  % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + m0(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz0(n) = tmp;
         
end

M0 = sparse(vp1, vp2, vz0, size(p,2), size(p,2));

%  calculate M1
[row1, col1] = find(m1);
vp1 = zeros(1, length(row1));
vp2 = zeros(1, length(row1));
vz1 = zeros(1, length(row1));

for n = 1:length(row1)
    
    p1 = row1(n);
    p2 = col1(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0 % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + m1(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz1(n) = tmp;
         
end

M1 = sparse(vp1, vp2, vz1, size(p,2), size(p,2));

%  calculate MDelta
[rowDelta, colDelta] = find(mDelta);
vp1 = zeros(1, length(rowDelta));
vp2 = zeros(1, length(rowDelta));
vz = zeros(1, length(rowDelta));

for n = 1:length(rowDelta)
    
    p1 = rowDelta(n);
    p2 = colDelta(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0 % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + mDelta(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz(n) = tmp;
         
end

MDelta = sparse(vp1, vp2, vz, size(p,2), size(p,2));
% [tm0, tm1] = fn_zhuang(p, e, t);

% 截断掉Hardwall边界上的点
M0 = M0(vinternal, vinternal);
M1 = M1(vinternal, vinternal);
MDelta = MDelta(vinternal, vinternal);

A = M1 + c*MDelta;
B = M0;

% [v, d] = eig(full(A), full(B));
% vep = diag(d);
% [~, vsort] = sort(real(vep));
% vep2 = vep(vsort);
% v2 = v(:, vsort);

[v, vep] = fn_SolveEigen(A,B);


cWaveFuncGamma1 = cell(1, length(vep));
for n = 1:length(vep)
    
    En = vep(n);
    vPsi = zeros(size(p,2), 1);
    vPsi(vinternal) = v(:,n);
    
    cWaveFuncGamma1{n} = {En, {vPsi}, {va}};
    
end

%%%%%%  完成计算 1维表示， 第1类

%%%%%%  1维表示， 第2类
va = [1,1,1,1,1,1,-1,-1,-1,-1,-1,-1];
va1 = va;
va2 = va;

%  calculate M0
[row0, col0] = find(m0);
vp1 = zeros(1, length(row0));
vp2 = zeros(1, length(row0));
vz0 = zeros(1, length(row0));

for n = 1:length(row0)
    
    p1 = row0(n);
    p2 = col0(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0  % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + m0(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz0(n) = tmp;
         
end

M0 = sparse(vp1, vp2, vz0, size(p,2), size(p,2));

%  calculate M1
[row1, col1] = find(m1);
vp1 = zeros(1, length(row1));
vp2 = zeros(1, length(row1));
vz1 = zeros(1, length(row1));

for n = 1:length(row1)
    
    p1 = row1(n);
    p2 = col1(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0 % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + m1(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz1(n) = tmp;
         
end

M1 = sparse(vp1, vp2, vz1, size(p,2), size(p,2));

%  calculate MDelta
[rowDelta, colDelta] = find(mDelta);
vp1 = zeros(1, length(rowDelta));
vp2 = zeros(1, length(rowDelta));
vz = zeros(1, length(rowDelta));

for n = 1:length(rowDelta)
    
    p1 = rowDelta(n);
    p2 = colDelta(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0 % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + mDelta(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz(n) = tmp;
         
end

MDelta = sparse(vp1, vp2, vz, size(p,2), size(p,2));
% [tm0, tm1] = fn_zhuang(p, e, t);

tv = find(abs(diag(M0))>1e-10);
vIndNonZero = intersect(vinternal, tv);
% 截断掉Hardwall边界上的点
M0 = M0(vIndNonZero, vIndNonZero);
M1 = M1(vIndNonZero, vIndNonZero);
MDelta = MDelta(vIndNonZero, vIndNonZero);

A = M1 + c*MDelta;
B = M0;

% [v, d] = eig(full(A), full(B));
% vep = diag(d);
% [~, vsort] = sort(real(vep));
% vep = vep(vsort);
% v = v(:, vsort);

[v, vep] = fn_SolveEigen(A,B);

cWaveFuncGamma2 = cell(1, length(vep));
for n = 1:length(vep)
    
    En = vep(n);
    vPsi = zeros(size(p,2), 1);
    vPsi(vIndNonZero) = v(:,n);
    
    cWaveFuncGamma2{n} = {En, {vPsi}, {va}};
    
end


%%%%%%  完成计算 1维表示， 第2类

%%%%%%  1维表示， 第3类
va = [1,-1,1,-1,1,-1,1,-1,1,-1,1,-1];
va1 = va;
va2 = va;

%  calculate M0
[row0, col0] = find(m0);
vp1 = zeros(1, length(row0));
vp2 = zeros(1, length(row0));
vz0 = zeros(1, length(row0));

for n = 1:length(row0)
    
    p1 = row0(n);
    p2 = col0(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0  % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + m0(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz0(n) = tmp;
         
end

M0 = sparse(vp1, vp2, vz0, size(p,2), size(p,2));

%  calculate M1
[row1, col1] = find(m1);
vp1 = zeros(1, length(row1));
vp2 = zeros(1, length(row1));
vz1 = zeros(1, length(row1));

for n = 1:length(row1)
    
    p1 = row1(n);
    p2 = col1(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0 % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + m1(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz1(n) = tmp;
         
end

M1 = sparse(vp1, vp2, vz1, size(p,2), size(p,2));

%  calculate MDelta
[rowDelta, colDelta] = find(mDelta);
vp1 = zeros(1, length(rowDelta));
vp2 = zeros(1, length(rowDelta));
vz = zeros(1, length(rowDelta));

for n = 1:length(rowDelta)
    
    p1 = rowDelta(n);
    p2 = colDelta(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0 % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + mDelta(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz(n) = tmp;
         
end

MDelta = sparse(vp1, vp2, vz, size(p,2), size(p,2));
% [tm0, tm1] = fn_zhuang(p, e, t);

tv = find(abs(diag(M0))>1e-10);
vIndNonZero = intersect(vinternal, tv);
% 截断掉Hardwall边界上的点
M0 = M0(vIndNonZero, vIndNonZero);
M1 = M1(vIndNonZero, vIndNonZero);
MDelta = MDelta(vIndNonZero, vIndNonZero);

A = M1 + c*MDelta;
B = M0;

% [v, d] = eig(full(A), full(B));
% vep = diag(d);
% [~, vsort] = sort(real(vep));
% vep = vep(vsort);
% v = v(:, vsort);

[v, vep] = fn_SolveEigen(A,B);

cWaveFuncGamma3 = cell(1, length(vep));
for n = 1:length(vep)
    
    En = vep(n);
    vPsi = zeros(size(p,2), 1);
    vPsi(vIndNonZero) = v(:,n);
    
    cWaveFuncGamma3{n} = {En, {vPsi}, {va}};
    
end


%%%%%%  完成计算 1维表示， 第3类

%%%%%%  1维表示， 第4类
va = [1,-1,1,-1,1,-1,-1,1,-1,1,-1,1];
va1 = va;
va2 = va;

%  calculate M0
[row0, col0] = find(m0);
vp1 = zeros(1, length(row0));
vp2 = zeros(1, length(row0));
vz0 = zeros(1, length(row0));

for n = 1:length(row0)
    
    p1 = row0(n);
    p2 = col0(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0  % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + m0(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz0(n) = tmp;
         
end

M0 = sparse(vp1, vp2, vz0, size(p,2), size(p,2));

%  calculate M1
[row1, col1] = find(m1);
vp1 = zeros(1, length(row1));
vp2 = zeros(1, length(row1));
vz1 = zeros(1, length(row1));

for n = 1:length(row1)
    
    p1 = row1(n);
    p2 = col1(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0 % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + m1(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz1(n) = tmp;
         
end

M1 = sparse(vp1, vp2, vz1, size(p,2), size(p,2));

%  calculate MDelta
[rowDelta, colDelta] = find(mDelta);
vp1 = zeros(1, length(rowDelta));
vp2 = zeros(1, length(rowDelta));
vz = zeros(1, length(rowDelta));

for n = 1:length(rowDelta)
    
    p1 = rowDelta(n);
    p2 = colDelta(n);
    
    switch vPointType(p1)
        case -1
            tm1 = mOneHotTop;
        case 0
            tm1 = mOneHotEdge0;
        case 1
            tm1 = mOneHotEdge1;
        case 2
            tm1 = mOneHotInt;
        otherwise
            error('wrong point type');
    end
    
     switch vPointType(p2)
        case -1
            tm2 = mOneHotTop;
        case 0
            tm2 = mOneHotEdge0;
        case 1
            tm2 = mOneHotEdge1;
        case 2
            tm2 = mOneHotInt;
        otherwise
            error('wrong point type');
     end
    
     tmp = 0;
     for ng1 = 1:12
         for ng2 = 1:12
             vOneHot1 = tm1(ng1,:);
             vOneHot2 = tm2(ng2,:);
             if sum(vOneHot1.*vOneHot2)~=0 % 判断两个群员作用后是否有交叠
                 tmp = tmp + va1(ng1)'*va2(ng2)*(0 + mDelta(p1, p2));
             end
         end
     end  
     
     vp1(n) = p1;
     vp2(n) = p2;
     vz(n) = tmp;
         
end

MDelta = sparse(vp1, vp2, vz, size(p,2), size(p,2));
% [tm0, tm1] = fn_zhuang(p, e, t);

tv = find(abs(diag(M0))>1e-10);
vIndNonZero = intersect(vinternal, tv);
% 截断掉Hardwall边界上的点
M0 = M0(vIndNonZero, vIndNonZero);
M1 = M1(vIndNonZero, vIndNonZero);
MDelta = MDelta(vIndNonZero, vIndNonZero);

A = M1 + c*MDelta;
B = M0;

% [v, d] = eig(full(A), full(B));
% vep = diag(d);
% [~, vsort] = sort(real(vep));
% vep = vep(vsort);
% v = v(:, vsort);

[v, vep] = fn_SolveEigen(A,B);

cWaveFuncGamma4 = cell(1, length(vep));
for n = 1:length(vep)
    
    En = vep(n);
    vPsi = zeros(size(p,2), 1);
    vPsi(vIndNonZero) = v(:,n);
    
    cWaveFuncGamma4{n} = {En, {vPsi}, {va}};
    
end

%%%%%%  完成计算 1维表示， 第4类

%%%%%% PART 6: 2D representation  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% PART 6.1: 二维表示计算的准备

V5X1 = [0,-1,-1,0,1,1,0,-1,-1,0,1,1]'/sqrt(8);
V5X2 = [2,1,-1,-2,-1,1,-2,-1,1,2,1,-1]'/sqrt(24);
V5Y1 = [2,1,-1,-2,-1,1,2,1,-1,-2,-1,1]'/sqrt(24);
V5Y2 = [0,1,1,0,-1,-1,0,-1,-1,0,1,1]'/sqrt(8);

V6X1 = [0,-1,1,0,-1,1,0,-1,1,0,-1,1]'/sqrt(8);
V6X2 = [2,-1,-1,2,-1,-1,-2,1,1,-2,1,1]'/sqrt(24);
V6Y1 = [2,-1,-1,2,-1,-1,2,-1,-1,2,-1,-1]'/sqrt(24);
V6Y2 = [0,1,-1,0,1,-1,0,-1,1,0,-1,1]'/sqrt(8);

V5A1 = [2,1,-1,-2,-1,1]'/sqrt(12);
V5A2 = [0,1,1,0,-1,-1]'/2;

V5B1 = [1,0,-1,-1,0,1]'/2;
V5B2 = [1,2,1,-1,-2,-1]'/sqrt(12);

V6A1 = [2,-1,-1,2,-1,-1]'/sqrt(12);
V6A2 = [0,1,-1,0,1,-1]'/2;

V6B1 = [1,-2,1,1,-2,1]'/sqrt(12);
V6B2 = [1,0,-1,1,0,-1]'/2;

Mii = eye(12);
Mia = [1,0,0,0,0,0; ...
    0,1,0,0,0,0; ...
    0,0,1,0,0,0; ...
    0,0,0,1,0,0; ...
    0,0,0,0,1,0; ...
    0,0,0,0,0,1; ...
    1,0,0,0,0,0; ...
    0,0,0,0,0,1; ...
    0,0,0,0,1,0; ...
    0,0,0,1,0,0; ...
    0,0,1,0,0,0; ...
    0,1,0,0,0,0];

Mib = [1,0,0,0,0,0; ...
    0,1,0,0,0,0; ...
    0,0,1,0,0,0; ...
    0,0,0,1,0,0; ...
    0,0,0,0,1,0; ...
    0,0,0,0,0,1; ...
    0,0,0,0,0,1; ...
    0,0,0,0,1,0; ...
    0,0,0,1,0,0; ...
    0,0,1,0,0,0; ...
    0,1,0,0,0,0; ...
    1,0,0,0,0,0];

Maa = eye(6);
Mbb = eye(6);
Mab = [1,0,0,0,0,1; ...
    1,1,0,0,0,0; ...
    0,1,1,0,0,0; ...
    0,0,1,1,0,0; ...
    0,0,0,1,1,0; ...
    0,0,0,0,1,1];

cMetric = cell(4,4);
cMetric{1,1} = Mii;
cMetric{1,2} = Mii;
cMetric{1,3} = Mia;
cMetric{1,4} = Mib;
cMetric{2,1} = Mii;
cMetric{2,2} = Mii;
cMetric{2,3} = Mia;
cMetric{2,4} = Mib;
cMetric{3,1} = Mia';
cMetric{3,2} = Mia';
cMetric{3,3} = Maa;
cMetric{3,4} = Mab;
cMetric{4,1} = Mib';
cMetric{4,2} = Mib';
cMetric{4,3} = Mab';
cMetric{4,4} = Mbb;

%%%%%% END PART 6.1 

%%%%%% PART 6.2:第一个表示空间的计算 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cI = cell(1,4);
cI{1} = vInternal;
cI{2} = vInternal;
cI{3} = vEdge1;
cI{4} = vEdge0;

cX = cell(1,4);
cX{1} = V5X1;
cX{2} = V5Y1;
cX{3} = V5A1;
cX{4} = V5B1;

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*m0(vI1, vI2);
        
    end
end

M0 = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*m1(vI1, vI2);
        
    end
end

M1 = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*mDelta(vI1, vI2);
        
    end
end

MDelta = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

A = M1 + c*MDelta;
B = M0;

% A = 1/2*(A + A');
% B = 1/2*(B + B');

[v, d] = eig(full(A), full(B));
vep = diag(d);
[~, vsort] = sort(real(vep));
vep = vep(vsort);
v = v(:,vsort);

cWaveFuncGamma5_1 = cell(1, length(vep));
for n = 1:length(vep)
    
    En = vep(n);
    tv = v(:,n);
    tL1 = length(vInternal);
    tL2 = length(vInternal);
    tL3 = length(vEdge1);
    tL4 = length(vEdge0);
    tmp1 = tv(1:tL1);
    tmp2 = tv((tL1+1):(tL1+tL2));
    tmp3 = tv((tL1+tL2+1):(tL1+tL2+tL3));
    tmp4 = tv((tL1+tL2+tL3+1):(tL1+tL2+tL3+tL4));
    vPsi1 = zeros(size(p,2),1);
    vPsi2 = zeros(size(p,2),1);
    vPsi3 = zeros(size(p,2),1);
    vPsi4 = zeros(size(p,2),1);
    
    vPsi1(vInternal) = tmp1;
    vPsi2(vInternal) = tmp2;
    vPsi3(vEdge1) = tmp3;
    vPsi4(vEdge0) = tmp4;
    
    cWaveFuncGamma5_1{n} = {En, {vPsi1, vPsi2, vPsi3, vPsi4}, ...
        {V5X1, V5Y1, V5A1, V5B1}};
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cI = cell(1,4);
cI{1} = vInternal;
cI{2} = vInternal;
cI{3} = vEdge1;
cI{4} = vEdge0;

cX = cell(1,4);
cX{1} = V5X2;
cX{2} = V5Y2;
cX{3} = V5A2;
cX{4} = V5B2;

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*m0(vI1, vI2);
        
    end
end

M0 = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*m1(vI1, vI2);
        
    end
end

M1 = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*mDelta(vI1, vI2);
        
    end
end

MDelta = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

A = M1 + c*MDelta;
B = M0;

% A = 1/2*(A + A');
% B = 1/2*(B + B');

% [v, d] = eig(full(A), full(B));
% vep = diag(d);
% [~, vsort] = sort(real(vep));
% vep = vep(vsort);
% v = v(:,vsort);

[v, vep] = fn_SolveEigen(A,B);

cWaveFuncGamma5_2 = cell(1, length(vep));
for n = 1:length(vep)
    
    En = vep(n);
    tv = v(:,n);
    tL1 = length(vInternal);
    tL2 = length(vInternal);
    tL3 = length(vEdge1);
    tL4 = length(vEdge0);
    tmp1 = tv(1:tL1);
    tmp2 = tv((tL1+1):(tL1+tL2));
    tmp3 = tv((tL1+tL2+1):(tL1+tL2+tL3));
    tmp4 = tv((tL1+tL2+tL3+1):(tL1+tL2+tL3+tL4));
    vPsi1 = zeros(size(p,2),1);
    vPsi2 = zeros(size(p,2),1);
    vPsi3 = zeros(size(p,2),1);
    vPsi4 = zeros(size(p,2),1);
    
    vPsi1(vInternal) = tmp1;
    vPsi2(vInternal) = tmp2;
    vPsi3(vEdge1) = tmp3;
    vPsi4(vEdge0) = tmp4;
    
    cWaveFuncGamma5_2{n} = {En, {vPsi1, vPsi2, vPsi3, vPsi4}, ...
        {V5X2, V5Y2, V5A2, V5B2}};
    
end

%%%%%% PART 6.3:第二个表示空间的计算 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

cI = cell(1,4);
cI{1} = vInternal;
cI{2} = vInternal;
cI{3} = vEdge1;
cI{4} = vEdge0;

cX = cell(1,4);
cX{1} = V6X1;
cX{2} = V6Y1;
cX{3} = V6A1;
cX{4} = V6B1;

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*m0(vI1, vI2);
        
    end
end

M0 = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*m1(vI1, vI2);
        
    end
end

M1 = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*mDelta(vI1, vI2);
        
    end
end

MDelta = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

A = M1 + c*MDelta;
B = M0;

% A = 1/2*(A + A');
% B = 1/2*(B + B');

% [v, d] = eig(full(A), full(B));
% vep = diag(d);
% [~, vsort] = sort(real(vep));
% vep = vep(vsort);
% v = v(:,vsort);

[v, vep] = fn_SolveEigen(A,B);

cWaveFuncGamma6_1 = cell(1, length(vep));
for n = 1:length(vep)
    
    En = vep(n);
    tv = v(:,n);
    tL1 = length(vInternal);
    tL2 = length(vInternal);
    tL3 = length(vEdge1);
    tL4 = length(vEdge0);
    tmp1 = tv(1:tL1);
    tmp2 = tv((tL1+1):(tL1+tL2));
    tmp3 = tv((tL1+tL2+1):(tL1+tL2+tL3));
    tmp4 = tv((tL1+tL2+tL3+1):(tL1+tL2+tL3+tL4));
    vPsi1 = zeros(size(p,2),1);
    vPsi2 = zeros(size(p,2),1);
    vPsi3 = zeros(size(p,2),1);
    vPsi4 = zeros(size(p,2),1);
    
    vPsi1(vInternal) = tmp1;
    vPsi2(vInternal) = tmp2;
    vPsi3(vEdge1) = tmp3;
    vPsi4(vEdge0) = tmp4;
    
    cWaveFuncGamma6_1{n} = {En, {vPsi1, vPsi2, vPsi3, vPsi4}, ...
        {V6X1, V6Y1, V6A1, V6B1}};
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cI = cell(1,4);
cI{1} = vInternal;
cI{2} = vInternal;
cI{3} = vEdge1;
cI{4} = vEdge0;

cX = cell(1,4);
cX{1} = V6X2;
cX{2} = V6Y2;
cX{3} = V6A2;
cX{4} = V6B2;

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*m0(vI1, vI2);
        
    end
end

M0 = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*m1(vI1, vI2);
        
    end
end

M1 = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

ct = cell(4,4);
for n1 = 1:4
    for n2 = 1:4
        
        vI1 = cI{n1};
        vI2 = cI{n2};
        vX1 = cX{n1};
        vX2 = cX{n2};
        Metric = cMetric{n1, n2};
    
        ct{n1, n2} = (vX1'*Metric*vX2)*mDelta(vI1, vI2);
        
    end
end

MDelta = [ct{1,1}, ct{1,2}, ct{1,3}, ct{1,4}; ...
    ct{2,1}, ct{2,2}, ct{2,3}, ct{2,4}; ...
    ct{3,1}, ct{3,2}, ct{3,3}, ct{3,4}; ...
    ct{4,1}, ct{4,2}, ct{4,3}, ct{4,4}];

A = M1 + c*MDelta;
B = M0;

% A = 1/2*(A + A');
% B = 1/2*(B + B');

% [v, d] = eig(full(A), full(B));
% vep = diag(d);
% [~, vsort] = sort(real(vep));
% vep = vep(vsort);
% v = v(:,vsort);

[v, vep] = fn_SolveEigen(A,B);

cWaveFuncGamma6_2 = cell(1, length(vep));
for n = 1:length(vep)
    
    En = vep(n);
    tv = v(:,n);
    tL1 = length(vInternal);
    tL2 = length(vInternal);
    tL3 = length(vEdge1);
    tL4 = length(vEdge0);
    tmp1 = tv(1:tL1);
    tmp2 = tv((tL1+1):(tL1+tL2));
    tmp3 = tv((tL1+tL2+1):(tL1+tL2+tL3));
    tmp4 = tv((tL1+tL2+tL3+1):(tL1+tL2+tL3+tL4));
    vPsi1 = zeros(size(p,2),1);
    vPsi2 = zeros(size(p,2),1);
    vPsi3 = zeros(size(p,2),1);
    vPsi4 = zeros(size(p,2),1);
    
    vPsi1(vInternal) = tmp1;
    vPsi2(vInternal) = tmp2;
    vPsi3(vEdge1) = tmp3;
    vPsi4(vEdge0) = tmp4;
    
    cWaveFuncGamma6_2{n} = {En, {vPsi1, vPsi2, vPsi3, vPsi4}, ...
        {V6X2, V6Y2, V6A2, V6B2}};
    
end

fprintf('finished Part 6\n');

%%%%%% END PART 6 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%   PART 7: 
%%%% 生成整个格子

cgMatrix = cell(1, 2*N);    % 每个群员的操作矩阵
mr = [cos(2*pi/N), -sin(2*pi/N); sin(2*pi/N), cos(2*pi/N)];
ms = [1,0;0,-1];

for n = 1:N
   cgMatrix{n} = mr^(n-1); 
end

for n = 1:N
   cgMatrix{N+n} = ms*mr^(n-1); 
end

p1 = p;
p2 = cgMatrix{2}*p;
p3 = cgMatrix{3}*p;
p4 = cgMatrix{4}*p;
p5 = cgMatrix{5}*p;
p6 = cgMatrix{6}*p;
p7 = cgMatrix{7}*p;
p8 = cgMatrix{8}*p;
p9 = cgMatrix{9}*p;
p10 = cgMatrix{10}*p;
p11 = cgMatrix{11}*p;
p12 = cgMatrix{12}*p;

t1 = t;
t2 = t;
t3 = t;
t4 = t;
t5 = t;
t6 = t;
t7 = t;
t8 = t;
t9 = t;
t10 = t;
t11 = t;
t12 = t;
pIndex1 = 1:size(p, 2);
pIndex2 = 1:size(p, 2);
pIndex3 = 1:size(p, 2);
pIndex4 = 1:size(p, 2);
pIndex5 = 1:size(p, 2);
pIndex6 = 1:size(p, 2);
pIndex7 = 1:size(p, 2);
pIndex8 = 1:size(p, 2);
pIndex9 = 1:size(p, 2);
pIndex10 = 1:size(p, 2);
pIndex11 = 1:size(p, 2);
pIndex12 = 1:size(p, 2);

p0 = p;
t0 = t;

p = p1;
t = t1;
pIndex = pIndex1;

%  通过combine方式贴合生成大格子
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p2, t2, pIndex2);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p3, t3, pIndex3);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p4, t4, pIndex4);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p5, t5, pIndex5);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p6, t6, pIndex6);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p7, t7, pIndex7);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p8, t8, pIndex8);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p9, t9, pIndex9);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p10, t10, pIndex10);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p11, t11, pIndex11);
[p, t, pIndex] = fn_combine_pt(p, t, pIndex, p12, t12, pIndex12);



N = length(cWaveFuncGamma1);
vEnGamma1 = zeros(N, 1);
for n = 1:N
   tc = cWaveFuncGamma1{n}; 
   vEnGamma1(n) =  tc{1};
end

N = length(cWaveFuncGamma2);
vEnGamma2 = zeros(N, 1);
for n = 1:N
   tc = cWaveFuncGamma2{n}; 
   vEnGamma2(n) =  tc{1};
end

N = length(cWaveFuncGamma3);
vEnGamma3 = zeros(N, 1);
for n = 1:N
   tc = cWaveFuncGamma3{n}; 
   vEnGamma3(n) =  tc{1};
end

N = length(cWaveFuncGamma4);
vEnGamma4 = zeros(N, 1);
for n = 1:N
   tc = cWaveFuncGamma4{n}; 
   vEnGamma4(n) =  tc{1};
end

N = length(cWaveFuncGamma5_1);
vEnGamma5_1 = zeros(N, 1);
for n = 1:N
   tc = cWaveFuncGamma5_1{n}; 
   vEnGamma5_1(n) =  tc{1};
end

N = length(cWaveFuncGamma5_2);
vEnGamma5_2 = zeros(N, 1);
for n = 1:N
   tc = cWaveFuncGamma5_2{n}; 
   vEnGamma5_2(n) =  tc{1};
end

N = length(cWaveFuncGamma6_1);
vEnGamma6_1 = zeros(N, 1);
for n = 1:N
   tc = cWaveFuncGamma6_1{n}; 
   vEnGamma6_1(n) =  tc{1};
end

N = length(cWaveFuncGamma6_2);
vEnGamma6_2 = zeros(N, 1);
for n = 1:N
   tc = cWaveFuncGamma6_2{n}; 
   vEnGamma6_2(n) =  tc{1};
end

vEnTotal = [vEnGamma1;vEnGamma2;vEnGamma3;vEnGamma4; ...
    vEnGamma5_1;vEnGamma5_2;vEnGamma6_1;vEnGamma6_2];
vType = [ones(size(vEnGamma1))*1;...
    ones(size(vEnGamma2))*2;...
    ones(size(vEnGamma3))*3;...
    ones(size(vEnGamma4))*4;...
    ones(size(vEnGamma5_1))*51;...
    ones(size(vEnGamma5_2))*52;...
    ones(size(vEnGamma6_1))*61;...
    ones(size(vEnGamma6_2))*62];
[~,vsort] = sort(vEnTotal);
vEnTotal = vEnTotal(vsort);
vType = vType(vsort);

cSolution = cWaveFuncGamma4{4};
vu = fn_Ext(p0, p, cSolution, cgMatrix);
En = cSolution{1};

% cSolution = cWaveFuncGamma4{4};
% vu = fn_Ext(p0, p, cSolution, cgMatrix);
% En = cSolution{1};
% 
% vuc = vu(vinternal);
% tv = A*vuc - En*B*vuc;
% disp(norm(tv));
% 
% figure(66);
% pdeplot(p, [], t, 'XYData', vu, 'ZDATA', vu, 'Contour', 'on', 'Mesh', 'off');
% colormap('jet');  % 设置颜色映射
% colorbar;  % 显示颜色条
% title('三角形区域上的解的三维分布');
% xlabel('X');
% ylabel('Y');
% zlabel('解 u');
% view(3);  % 设置为三维视角
% axis equal;  % 坐标轴等比例显示
% X = vu;
% 
% triang = triangulation(t(1:3,:)', p(1,:)', p(2,:)');
% vtheta =linspace(0, 2*pi, 100);
% vr = zeros(1, length(vtheta));
% for itheta = 1:length(vtheta)
% 
%     
%     th = vtheta(itheta);
%     x0 = 1/2*cos(th);
%     y0 = 1/2*sin(th);
%     queryPoint = [x0, y0];
%     tid = pointLocation(triang, queryPoint);
%     vertices = t(1:3, tid);
%     
%     % 提取三角形顶点的坐标和场值
%     triCoords = [p(1, vertices)', p(2, vertices)']; % 三角形顶点坐标
%     triValues = X(vertices); % 三角形顶点场值
%     
%     % 计算查询点的重心坐标
%     baryCoords = cartesianToBarycentric(triang, tid, queryPoint);
%     interpValue = baryCoords * triValues;
%     
%     vr(itheta) = interpValue;
%     
% end
% 
% figure(9);
% plot(vtheta, vr, '-bo');
%         

disp('pwcrfxcl');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('\n ============================ \n');
% fprintf('begin to search eigs\n');
% 
% search_step = 20;
% search_eig_min = 10; % initial
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
% 
% EnGs = rv(1);
% vPhiGsInt = rm(:,1);
% vPhiGs = zeros(size(p0,2), 1);
% vPhiGs(vinternal) = vPhiGsInt;
% 
% 
% close(figure(5));
% figure(5);
% x = p0(1, :);  % x 坐标
% y = p0(2, :);  % y 坐标
% tri = t0(1:3, :)';  % 转置为 Nt x 3 的矩阵
% 
% % 绘制三维曲面
% % trisurf(tri, x, y, vPhiGs, 'EdgeColor', 'none');  % 绘制三角形曲面
% pdeplot(p0, [], t0, 'XYData', vPhiGs, 'ZDATA', vPhiGs, 'Contour', 'on', 'Mesh', 'off');
% colormap('jet');  % 设置颜色映射
% colorbar;  % 显示颜色条
% title('三角形区域上的解的三维分布');
% xlabel('X');
% ylabel('Y');
% zlabel('解 u');
% % view(3);  % 设置为三维视角
% axis equal;  % 坐标轴等比例显示
% 
% vPhiGsTotal = zeros(size(p,2), 1);
% 
% for n = 1:length(va)
%     for nPoint = 1:size(p0,2)
%         vxy = cgMatrix{n}*p0(:,nPoint);
%         vDiss = sum((vxy - p).^2);
%         
%         IndexFind = find(vDiss<1e-11);
%         if length(IndexFind)~=1
%             error('wrong grid');
%         end
%         
%         vPhiGsTotal(IndexFind) = vPhiGsTotal(IndexFind) + va(n)*vPhiGs(nPoint);
%         
%     end
% end
% 
% figure(6);
% 
% pdeplot(p, [], t, 'XYData', vPhiGsTotal, 'Contour', 'on', 'Mesh', 'off');
% colormap('jet');  % 设置颜色映射
% colorbar;  % 显示颜色条
% title('三角形区域上的解的三维分布');
% xlabel('X');
% ylabel('Y');
% zlabel('解 u');
% view(2);  % 设置为三维视角
% axis equal;  % 坐标轴等比例显示

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


function [rm, rv] = fn_SolveEigen(A, B)

fprintf('\n ============================ \n');
fprintf('begin to search eigs\n');

search_step = 50;
search_eig_min = 0; % initial
region_begin = search_eig_min;

num_step = 10;
c_lmb = cell(1,num_step);
c_xv = cell(1,num_step);

for istep = 1:num_step
     
    while 1
        search_eig_max = search_eig_min + search_step;
        [xv,lmb,iresult]...
            = sptarn(A,B,search_eig_min,search_eig_max,1,100*eps,100);

        if iresult>=0
            break;
        else
            search_step = search_step/2;
        end

    end
    
    c_lmb{istep} = lmb;
    c_xv{istep} = xv;
    fprintf('=======================================\n');
    fprintf('finished interval [%4.2f, %4.2f], find eigs %d \n',...
        search_eig_min,search_eig_max,iresult);
  
    search_eig_min = search_eig_max;
end

[rv,rm] = fn_cell_2_vec(c_lmb,c_xv);
rv = rv';


end
