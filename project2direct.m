% x, y coordination
%xy=csvread('SiouxFalls_node.csv');

% All roads -> type 1

%Sioux Falls City Road Network Data - capacity
road=csvread('net_EMA.csv');
matrix=csvread('matrix_EMA.csv');

% Sioux Falls City Road Network Data - length
%road=csvread('net_length.csv');

%matrix=csvread('matrix_length.csv');

s = road(1:258,1);
t = road(1:258,2);
w = road(1:258,3);

%x = xy(1:24,2);
%y = xy(1:24,3);

G = digraph(s,t,w);
h = plot(G);


%matrix=1./matrix;
matrix(matrix==0)=inf;



distance = 0;
distance_neg = 0;
dis_neg_list = zeros(24);
for i = 1:73
    for j = i+1:74
        [mydistance,mypath]=mydijkstra(matrix,i,j);
        distance = mydistance + distance;
        distance_neg = 1/mydistance + distance_neg;
    end
    dis_neg_list(i)=1/mydistance;
end
csvwrite('m_wcc.csv',dis_neg_list);
% Dijkstra algorithm
function [mydistance,mypath]=mydijkstra(a,sb,db);
    % insert: a:matrix, sb: start basepoint, db: depart basepoint
    % output: mydistance: shortest length, mypath: the shortest path
n=size(a,1); visited(1:n) = 0;
distance(1:n) = inf; % save the shortest path from sb to all other points
distance(sb) = 0; parent(1:n) = 0;
for i = 1: n-1
    temp=distance;
    id1=find(visited==1); 
    temp(id1)=inf; 
    [t, u] = min(temp); 
    visited(u) = 1; 
    id2=find(visited==0); 
    for v = id2
        if a(u, v) + distance(u) < distance(v)
            distance(v) = distance(u) + a(u, v); 
            parent(v) = u;
        end
    end
end
mypath = [];
if parent(db) ~= 0 
    t = db; mypath = [db];
while t ~= sb
    p = parent(t);
    mypath = [p mypath];
    t = p;
end
end
mydistance = distance(db);
return
end



