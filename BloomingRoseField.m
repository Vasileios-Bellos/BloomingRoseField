% BloomingRoseField.m — Blooming Rose Field with Hovering Light Source
%
% A field of roses spread across a ground plane. A glowing light source
% hovers above the field. Roses within the light's influence radius bloom;
% those outside gradually close back up.
%
% THREE CONTROL MODES (press 1/2/3 to Switch at any time):
%   1 — Mouse:   Light follows cursor with smooth interpolation (default)
%   2 — Arrows:  Arrow keys move the light (hold for continuous motion)
%   3 — Auto:    Lissajous figure-8 path sweeping the field
%
% Other keys: Space = Pause/Unpause, Q/X/Escape = Quit
%
% Transitions are seamless — the new mode starts from the light's current
% position, so there's no jump when switching.
%
% PERFORMANCE: Bloom levels are precomputed. Per-rose transforms are cached.
% Only roses whose integer bloom level changed get a surf update each frame.
%
% Rose head geometry adapted from Eric Ludlam's "Blooming Rose" (MATLAB Flipbook Mini Hack, 2023)
%
% Ref: https://uk.mathworks.com/matlabcentral/communitycontests/contests/6/entries/13857
%      https://github.com/zappo2/digital-art-with-matlab/tree/master/flowers
%      https://github.com/Vasileios-Bellos/BloomingRose
%
% Vasilis Bellos, 2026

%% ========== PARAMETERS ==========

% Playback
referenceFPS  = 30;                   % target frame rate (numeric = real-time, 'fixed' = constant timestep)

% Roses
nRoses       = 15;                    % number of roses in the field
n            = 100;                   % mesh resolution (n × n grid per rose)
A_rose       = 1.995653;              % petal height coefficient
B_rose       = 1.27689;               % petal curl coefficient
petalNum     = 3.6;                   % number of petals per revolution
nBloomLevels = 60;                    % discrete bloom states to precompute
roseScale    = 0.5;                   % base rose size (multiplied by per-rose variation)

% Bloom dynamics
bloomRate    = 3.0;                   % bloom speed when inside influence radius
unbloomRate  = 1.5;                   % unbloom speed when outside influence radius

% Sepals
nSepals      = 5;                     % number of sepals per rose
sepalLength  = 0.26;                  % tip-to-base length
sepalWidth   = 0.08;                  % max width at midpoint
sepalDroop   = 0.08;                  % outward droop
sepalColor   = [0.22 0.50 0.18];      % slightly brighter green

% Stems
stemLength    = 2.8;                  % total stem length
stemRadiusTop = 0.032;                % radius near calyx
stemRadiusBot = 0.026;                % radius at base
nStemLen      = 20;                   % segments along stem
nStemCirc     = 10;                   % segments around stem
stemLeanMax   = 0.12;                 % max lateral lean of top (fraction of height)
stemBowMax    = 0.18;                 % max mid-stem bow outward (fraction of height)
stemColor     = [0.18 0.42 0.15];     % dark green

% Thorns
thornsPerStem = 4;                    % base thorn count (actual varies ±1 per stem)
thornHeight   = 0.06;                 % cone height
thornRadius   = 0.015;                % cone base radius
thornColor    = [0.30 0.25 0.12];     % brownish-green

% Colors
groundColor   = [0.20 0.35 0.12];     % ground plane colour

% Light source
lightHeight     = 2.5;                % hover height above ground
influenceRadius = lightHeight - 0.5;  % bloom trigger radius
lightColor      = [1.0 0.95 0.6];     % warm yellow-white
influenceAlpha  = 0.08;               % transparency of influence sphere

%% ========== COLORMAP LIBRARY ==========
clib = struct('name',{}, 'map',{});
clib(end+1) = struct('name','Classic red',   'map',[linspace(.25,1,256)',  linspace(0,.08,256)',   linspace(.02,.05,256)']);
clib(end+1) = struct('name','Juliet',        'map',[linspace(.55,1,256)',  linspace(.22,.72,256)', linspace(.10,.50,256)']);
clib(end+1) = struct('name','Amnesia',       'map',[linspace(.35,.76,256)',linspace(.28,.58,256)', linspace(.38,.64,256)']);
clib(end+1) = struct('name','Quicksand',     'map',[linspace(.45,.90,256)',linspace(.32,.72,256)', linspace(.28,.62,256)']);
clib(end+1) = struct('name','Sahara',        'map',[linspace(.50,.95,256)',linspace(.38,.82,256)', linspace(.18,.55,256)']);
clib(end+1) = struct('name','Coral reef',    'map',[linspace(.45,.98,256)',linspace(.12,.52,256)', linspace(.10,.45,256)']);
clib(end+1) = struct('name','Hot pink',      'map',[linspace(.35,1,256)',  linspace(.02,.28,256)', linspace(.18,.52,256)']);
clib(end+1) = struct('name','Blush',         'map',[linspace(.55,.96,256)',linspace(.35,.75,256)', linspace(.38,.76,256)']);
clib(end+1) = struct('name','Ocean Song',    'map',[linspace(.28,.68,256)',linspace(.18,.52,256)', linspace(.42,.78,256)']);
clib(end+1) = struct('name','Golden Mustard','map',[linspace(.45,.95,256)',linspace(.28,.75,256)', linspace(.02,.12,256)']);
clib(end+1) = struct('name','Ivory',         'map',[linspace(.65,1,256)',  linspace(.58,.96,256)', linspace(.45,.88,256)']);
clib(end+1) = struct('name','Free Spirit',   'map',[linspace(.50,1,256)',  linspace(.15,.55,256)', linspace(.02,.12,256)']);
clib(end+1) = struct('name','Burgundy',      'map',[linspace(.12,.50,256)',linspace(.02,.05,256)', linspace(.06,.15,256)']);
clib(end+1) = struct('name','Rose gold',     'map',[linspace(.42,.92,256)',linspace(.22,.58,256)', linspace(.18,.48,256)']);
clib(end+1) = struct('name','White Mondial', 'map',[linspace(.60,1,256)',  linspace(.68,1,256)',   linspace(.55,.95,256)']);
clib(end+1) = struct('name','Mint green',    'map',[linspace(.10,.85,256)',linspace(.35,1,256)',   linspace(.25,.75,256)']);
clib(end+1) = struct('name','Black Baccara', 'map',[linspace(.08,.55,256)',linspace(.01,.02,256)', linspace(.03,.06,256)']);
clib(end+1) = struct('name','Cafe Latte',    'map',[linspace(.25,.75,256)',linspace(.15,.58,256)', linspace(.08,.42,256)']);
clib(end+1) = struct('name','Aobara',        'map',[linspace(.12,.72,256)',linspace(.05,.45,256)', linspace(.28,.82,256)']);

%% ========== PALETTE — pick random colors ==========
rng('shuffle');
sel = randi(numel(clib), 1, nRoses);
roseMaps = clib(sel);

%% ========== FIELD LAYOUT (uniform trapezoidal placement) ==========

% Field geometry — front = near camera (+Y), back = far from camera (-Y).
groundZ      = -0.10;                % ground plane Z offset
fieldFrontY  = -4;                   % near camera (bottom of screen)
fieldBackY   = 4;                    % far from camera (top of screen)
fieldFrontW  = 5.5;                  % full width at front edge
fieldBackW   = 11;                   % full width at back edge
plotTrapz    = false;                % draw red trapezoid outline (debugging)

% Auto-compute separation so nRoses fill the field snugly
trapArea = (fieldFrontW + fieldBackW) * (fieldBackY - fieldFrontY) / 2;
minSep   = 0.60 * sqrt(trapArea / nRoses);

% Derived params that scale with the field
lightRadius     = sqrt(trapArea) / 80;   % visual size of light sphere
lightSpeed      = sqrt(trapArea) / 80;   % Lissajous angular velocity (radians/tick)
arrowSpeed      = sqrt(trapArea) / 10;   % arrow key movement speed
mouseLerp       = sqrt(trapArea) / 20;   % mouse smoothing rate

% 3-zone placement: split depth into 3 equal bands, place nRoses/3 in each.
% Front zones are narrower (trapezoid), so equal count = higher density near camera.
widthAt = @(y) fieldFrontW + (y - fieldFrontY) / (fieldBackY - fieldFrontY) * (fieldBackW - fieldFrontW);

fieldDepth = fieldBackY - fieldFrontY;
zone1Y = fieldFrontY + fieldDepth * 0.25;  % front 25% (near camera)
zone2Y = fieldFrontY + fieldDepth * 0.65;  % middle 40%

perZone = [round(nRoses/3), round(nRoses/3), nRoses - 2*round(nRoses/3)];
zoneBounds = [fieldFrontY, zone1Y;    % zone 1: front (near camera)
              zone1Y,      zone2Y;    % zone 2: middle
              zone2Y,      fieldBackY]; % zone 3: back (far from camera)

positions = zeros(nRoses, 2);
placed = 0;
maxAtt = 50000;

for zi = 1:3
    target = placed + perZone(zi);
    attempts = 0;
    while placed < target && attempts < maxAtt
        cy = zoneBounds(zi,1) + rand * (zoneBounds(zi,2) - zoneBounds(zi,1));
        w = widthAt(cy);
        cx = (rand - 0.5) * w;
        if placed == 0 || all(sqrt(sum((positions(1:placed,:) - [cx, cy]).^2, 2)) > minSep)
            placed = placed + 1;
            positions(placed,:) = [cx, cy];
        end
        attempts = attempts + 1;
    end
end
if placed < nRoses
    warning('Only placed %d / %d roses (increase field area or decrease minSep).', placed, nRoses);
    nRoses = placed;
    roseMaps = roseMaps(1:nRoses);
end

% Per-rose variation
roseScales = roseScale * (0.85 + 0.30 * rand(nRoses, 1));
roseRotZ   = rand(nRoses, 1) * 2 * pi;    % random Z rotation (petal orientation)

fprintf('Flower field: %d roses, minSep=%.1f, area=%.0f (Y: %+.0f to %+.0f, W: %.0f to %.0f).\n', ...
    nRoses, minSep, trapArea, fieldBackY, fieldFrontY, fieldFrontW, fieldBackW);

%% ========== PRECOMPUTE ALL BLOOM LEVELS ==========
fprintf('Precomputing %d bloom levels at resolution %d...\n', nBloomLevels, n);
wb = waitbar(0, 'Precomputing bloom levels...', 'Name', 'Flower Field');
cleanupWB = onCleanup(@() delete(wb(isvalid(wb))));

r     = linspace(0, 1, n);
theta = linspace(-2, 20*pi, n);
[RR, THETA] = ndgrid(r, theta);
xPetal = 1 - (1/2)*((5/4)*(1 - mod(petalNum*THETA, 2*pi)/pi).^2 - 1/4).^2;

f_norm     = linspace(1, 48, nBloomLevels);
openness   = 1.05 - cospi(f_norm/(48/2.5)) .* (1 - f_norm/48).^2;
opencenter = openness * 0.2;

bloomX = cell(nBloomLevels, 1);
bloomY = cell(nBloomLevels, 1);
bloomZ = cell(nBloomLevels, 1);

for bi = 1:nBloomLevels
    phi_k = (pi/2) * linspace(opencenter(bi), openness(bi), n).^2;
    y_k   = A_rose*(RR.^2).*(B_rose*RR-1).^2 .* sin(phi_k);
    R2_k  = xPetal .* (RR.*sin(phi_k) + y_k.*cos(phi_k));
    bloomX{bi} = R2_k .* sin(THETA);
    bloomY{bi} = R2_k .* cos(THETA);
    bloomZ{bi} = xPetal .* (RR.*cos(phi_k) - y_k.*sin(phi_k));

    if mod(bi, 10) == 0
        waitbar(bi / nBloomLevels, wb, sprintf('Bloom level %d / %d', bi, nBloomLevels));
    end
end

% Precompute CData for each rose (color depends on rose, not bloom level)
roseCData = cell(nRoses, 1);
mid = round(nBloomLevels / 2);
for ri = 1:nRoses
    roseCData{ri} = scalarToRGB(bloomX{mid}, bloomY{mid}, bloomZ{mid}, roseMaps(ri).map);
end

delete(wb);  clear wb cleanupWB
fprintf('Bloom levels ready.\n');

%% ========== SEPAL TEMPLATE ==========
nSu = 10;  nSv = 6;
u_sep = linspace(0, 1, nSu)';
v_sep = linspace(-1, 1, nSv);
sepalWP   = sepalWidth * sin(pi * u_sep).^0.6 .* (1 - u_sep.^3);
xLocal_s  = sepalWP .* v_sep;
zLocal_s  = sepalLength * u_sep .* (1 - 0.5*u_sep) + sepalDroop * u_sep.^2;
rLocal_s  = stemRadiusTop * (1 - u_sep*0.3) + sepalLength * 0.4 * u_sep.^1.5;
cupFactor = 0.02 * (1 - v_sep.^2);

%% ========== THORN CONE TEMPLATE ==========
nTu = 8;  nTv = 10;
[Uth, Vth] = meshgrid(linspace(0,1,nTu), linspace(0,2*pi,nTv));
R_cone = thornRadius * (1 - Uth).^1.5;
X_cone = R_cone .* cos(Vth);
Y_cone = R_cone .* sin(Vth);
Z_cone = thornHeight * Uth;

%% ========== BUILD STEMS (Bezier curves with random lean & bow) ==========
% Each stem is a cubic Bezier from ground to rose head. Random lean gives
% the top lateral offset from directly above the base. Random mid-stem bow
% creates gentle S-curves and outward arcs. Frenet frames on the spine
% give proper tube cross-sections. The stem-top tangent determines rose
% head orientation (no independent random tilt needed).
t_bez    = linspace(0, 1, nStemLen)';
phi_circ = linspace(0, 2*pi, nStemCirc);
r_prof   = stemRadiusTop + (stemRadiusBot - stemRadiusTop) * t_bez;
r_prof   = r_prof + 0.015 * exp(-((t_bez)/0.06).^2);

allStems    = struct('X',cell(1,nRoses), 'Y',cell(1,nRoses), 'Z',cell(1,nRoses));
allSepals   = cell(1, nRoses);
allThorns   = cell(1, nRoses);
stemTopPos  = zeros(nRoses, 3);     % where the rose head sits
stemTopTang = zeros(nRoses, 3);     % tangent at stem top → head orientation

for ri = 1:nRoses
    base_x = positions(ri,1);
    base_y = positions(ri,2);
    sc     = roseScales(ri);
    h      = stemLength * sc;

    % --- Random lean: stem top is laterally offset from base ---
    leanMag = h * stemLeanMax * (0.2 + 0.8*rand);
    leanAng = rand * 2*pi;
    lean_dx = leanMag * cos(leanAng);
    lean_dy = leanMag * sin(leanAng);

    % --- Random mid-stem bow (roughly perpendicular for S-shapes) ---
    bowMag  = h * stemBowMax * (0.15 + 0.85*rand);
    bowAng  = leanAng + pi/2 + (rand - 0.5) * pi * 0.7;  % mostly perpendicular
    bow_dx  = bowMag * cos(bowAng);
    bow_dy  = bowMag * sin(bowAng);

    % Cubic Bezier control points: P0 (base) → P3 (top)
    % Stem starts vertical from ground (P0→P1), then curves laterally
    % in the mid-section (P1→P2), and eases into the rose head (P2→P3).
    % Matches the proven pattern from RoseAnimated_stem.m.
    P0 = [base_x,              base_y,              groundZ];
    P3 = [base_x + lean_dx,   base_y + lean_dy,    groundZ + h];
    P1 = [base_x,             base_y,              groundZ + h * 0.35];  % vertical rise
    P2 = [base_x + lean_dx + bow_dx, base_y + lean_dy + bow_dy, groundZ + h * 0.65];  % lateral curve

    % Evaluate Bezier spine and tangent
    tt = t_bez;  tt1 = 1 - tt;
    spine = tt1.^3.*P0 + 3*tt1.^2.*tt.*P1 + 3*tt1.*tt.^2.*P2 + tt.^3.*P3;
    tang  = 3*tt1.^2.*(P1-P0) + 6*tt1.*tt.*(P2-P1) + 3*tt.^2.*(P3-P2);
    tang  = tang ./ max(vecnorm(tang, 2, 2), 1e-10);

    % Frenet frame: reference-vector method for stable normals
    refVec = [1, 0, 0];
    nrm  = cross(tang, repmat(refVec, nStemLen, 1), 2);
    nrmN = vecnorm(nrm, 2, 2);
    bad  = nrmN < 1e-6;
    if any(bad)
        nrm(bad,:) = cross(tang(bad,:), repmat([0 1 0], sum(bad), 1), 2);
        nrmN(bad)  = vecnorm(nrm(bad,:), 2, 2);
    end
    nrm  = nrm ./ max(nrmN, 1e-10);
    bnrm = cross(tang, nrm, 2);
    bnrm = bnrm ./ max(vecnorm(bnrm, 2, 2), 1e-10);

    % Build tube mesh
    rp = r_prof * (0.90 + 0.20*rand);
    Xs = zeros(nStemLen, nStemCirc);
    Ys = zeros(nStemLen, nStemCirc);
    Zs = zeros(nStemLen, nStemCirc);
    for si = 1:nStemLen
        for sj = 1:nStemCirc
            off = rp(si) * (nrm(si,:)*cos(phi_circ(sj)) + bnrm(si,:)*sin(phi_circ(sj)));
            Xs(si,sj) = spine(si,1) + off(1);
            Ys(si,sj) = spine(si,2) + off(2);
            Zs(si,sj) = spine(si,3) + off(3);
        end
    end
    allStems(ri).X = Xs;  allStems(ri).Y = Ys;  allStems(ri).Z = Zs;

    % Record stem-top position and tangent
    stemTopPos(ri,:)  = P3;
    stemTopTang(ri,:) = tang(end,:);

    % --- Sepals oriented by stem-top tangent ---
    Rm = buildHeadRotation(stemTopTang(ri,:), roseRotZ(ri));
    sepArr = struct('X',cell(1,nSepals), 'Y',cell(1,nSepals), 'Z',cell(1,nSepals));
    for s = 1:nSepals
        ang = (s-1)*2*pi/nSepals + pi/10 + rand*0.3;
        Xsep = rLocal_s .* cos(ang) + xLocal_s * (-sin(ang));
        Ysep = rLocal_s .* sin(ang) + xLocal_s * cos(ang);
        Zsep = zLocal_s + cupFactor .* u_sep;
        pts  = [Xsep(:)*sc, Ysep(:)*sc, Zsep(:)*sc] * Rm';
        sepArr(s).X = reshape(pts(:,1), nSu, nSv) + P3(1);
        sepArr(s).Y = reshape(pts(:,2), nSu, nSv) + P3(2);
        sepArr(s).Z = reshape(pts(:,3), nSu, nSv) + P3(3);
    end
    allSepals{ri} = sepArr;

    % --- Thorns along stem (from RoseAnimated_stem.m) ---
    nTh = thornsPerStem + randi([-1, 1]);  nTh = max(1, nTh);
    thornPos = linspace(0.12, 0.85, nTh) + 0.04*(rand(1, nTh)-0.5);
    thornPos = max(0.10, min(0.90, thornPos));
    thornAng = rand*2*pi + linspace(0, 2*pi, nTh+1);
    thornAng(end) = [];
    thornAng = thornAng + 0.3*(rand(1, nTh)-0.5);

    thArr = struct('X',cell(1,nTh), 'Y',cell(1,nTh), 'Z',cell(1,nTh));
    for th = 1:nTh
        idx = round(thornPos(th) * (nStemLen-1)) + 1;
        idx = max(1, min(nStemLen, idx));
        bPos  = spine(idx,:);
        T_vec = tang(idx,:);
        N_vec = nrm(idx,:);
        B_vec = bnrm(idx,:);

        outDir = N_vec*cos(thornAng(th)) + B_vec*sin(thornAng(th));
        thAxis = outDir*cos(pi/6) + T_vec*sin(pi/6);
        thAxis = thAxis / norm(thAxis);

        if abs(dot(thAxis,[1,0,0])) < 0.9, pRef=[1,0,0]; else, pRef=[0,1,0]; end
        thN = cross(thAxis, pRef); thN = thN/norm(thN);
        thB = cross(thAxis, thN);  thB = thB/norm(thB);

        szFactor = 0.85 + 0.30*rand;
        Xt = zeros(size(X_cone)); Yt = Xt; Zt = Xt;
        for ii = 1:numel(X_cone)
            lp = szFactor * (X_cone(ii)*thN + Y_cone(ii)*thB + Z_cone(ii)*thAxis);
            wp = bPos + rp(idx)*outDir + lp;
            Xt(ii) = wp(1); Yt(ii) = wp(2); Zt(ii) = wp(3);
        end
        thArr(th).X = Xt; thArr(th).Y = Yt; thArr(th).Z = Zt;
    end
    allThorns{ri} = thArr;
end

%% ========== PRECOMPUTE PER-ROSE TRANSFORMS ==========
% Constant for the entire animation. Each rose's scaled rotation matrix
% and world offset are used in the hot loop — precomputing avoids repeated
% cell/array lookups and redundant scale multiplies each frame.
roseScRm   = cell(nRoses, 1);     % sc * Rm' (3×3) for one matrix multiply
roseOffset = zeros(nRoses, 3);    % world position of rose head
rosePos3D  = zeros(nRoses, 3);    % same, for vectorized distance computation
for ri = 1:nRoses
    Rm = buildHeadRotation(stemTopTang(ri,:), roseRotZ(ri));
    roseScRm{ri}    = roseScales(ri) * Rm';
    roseOffset(ri,:) = stemTopPos(ri,:);
    rosePos3D(ri,:)  = stemTopPos(ri,:);
end

%% ========== LIGHT SPHERE GEOMETRY ==========
[LS_X, LS_Y, LS_Z] = sphere(24);
lsX = LS_X * lightRadius;
lsY = LS_Y * lightRadius;
lsZ = LS_Z * lightRadius;
% Smoother resolution for influence sphere
[LI_X, LI_Y, LI_Z] = sphere(32);
liX = LI_X * influenceRadius;
liY = LI_Y * influenceRadius;
liZ = LI_Z * influenceRadius;

%% ========== GROUND PLANE ==========
gE = max(fieldBackW/2, abs(fieldBackY)) + 6;   % ground mesh — covers visible area
[GX, GY] = meshgrid(linspace(-gE, gE, 50), linspace(-gE, gE, 50));
GZ = ones(size(GX)) * groundZ;

%% ========== FIGURE SETUP ==========
close all
fig = figure('Color', 'k', 'MenuBar', 'none', 'WindowStyle', 'normal', 'WindowState', 'maximized', 'Name', 'Blooming Rose Field', 'NumberTitle', 'off');
ax = axes('Parent', fig, 'Units', 'normalized', 'Position', [0 0 1 1], 'Nextplot', 'add', 'Clipping', 'off', 'Projection', 'perspective', 'Visible', 'off', 'DataAspectRatio',[1 1 1]);

% Ground
surf(ax, GX, GY, GZ, 'FaceColor', groundColor, 'EdgeColor', 'none', ...
    'FaceLighting','gouraud','AmbientStrength',0.4,'DiffuseStrength',0.6, ...
    'SpecularStrength',0.0);

if plotTrapz
    trapX = [-fieldFrontW/2, fieldFrontW/2, fieldBackW/2, -fieldBackW/2, -fieldFrontW/2];
    trapY = [fieldFrontY, fieldFrontY, fieldBackY, fieldBackY, fieldFrontY];
    trapZ = ones(1,5) * (groundZ + 0.02);
    line(ax, trapX, trapY, trapZ, 'Color', 'r', 'LineWidth', 2);
end

% Stems, sepals, and thorns
for ri = 1:nRoses
    surf(ax, allStems(ri).X, allStems(ri).Y, allStems(ri).Z, ...
        'FaceColor',stemColor,'EdgeColor','none','FaceLighting','gouraud', ...
        'AmbientStrength',0.4,'DiffuseStrength',0.7,'SpecularStrength',0.1);
    seps = allSepals{ri};
    for s = 1:nSepals
        surf(ax, seps(s).X, seps(s).Y, seps(s).Z, 'FaceColor',sepalColor, ...
            'EdgeColor','none','FaceLighting','gouraud','AmbientStrength',0.4, ...
            'DiffuseStrength',0.7,'BackFaceLighting','lit');
    end
    ths = allThorns{ri};
    for th = 1:numel(ths)
        surf(ax, ths(th).X, ths(th).Y, ths(th).Z, 'FaceColor',thornColor, ...
            'EdgeColor','none','FaceLighting','gouraud','AmbientStrength',0.3);
    end
end

% Rose head surfaces (start at bloom level 1 = closed bud)
hRoses = gobjects(nRoses, 1);
for ri = 1:nRoses
    pts = [bloomX{1}(:), bloomY{1}(:), bloomZ{1}(:)] * roseScRm{ri};
    Xw = reshape(pts(:,1), n, n) + roseOffset(ri,1);
    Yw = reshape(pts(:,2), n, n) + roseOffset(ri,2);
    Zw = reshape(pts(:,3), n, n) + roseOffset(ri,3);
    hRoses(ri) = surf(ax, Xw, Yw, Zw, roseCData{ri}, ...
        'LineStyle','none','FaceColor','interp','FaceLighting','gouraud', ...
        'AmbientStrength',0.3,'DiffuseStrength',0.7,'SpecularStrength',0.2, ...
        'BackFaceLighting','lit');
end

% Light source spheres
lightPos0 = [0, 0, groundZ + 2.5];
hLightSphere = surf(ax, lsX + lightPos0(1), lsY + lightPos0(2), lsZ + lightPos0(3), ...
    'FaceColor', lightColor, 'EdgeColor', 'none', ...
    'FaceLighting', 'none','AmbientStrength',1.0);
hInfluenceSphere = surf(ax, liX + lightPos0(1), liY + lightPos0(2), liZ + lightPos0(3), ...
    'FaceColor', lightColor, 'EdgeColor', 'none', ...
    'FaceAlpha', influenceAlpha, 'FaceLighting', 'none');

% MATLAB light that follows the sphere (actual illumination)
hMovingLight = light(ax, 'Position', lightPos0, 'Style', 'local', ...
    'Color', lightColor * 0.8);

% Subtle ambient lighting
light(ax, 'Position', [0 0 10], 'Style', 'infinite', 'Color', [0.15 0.15 0.25]);

% Camera: perspective, low angle
el = 40;  d = 12;
% Camera: shifted to to the base of the trapezoid
ax.CameraPosition  = [0, -d*cosd(el) - 3, groundZ + d*sind(el)];
ax.CameraTarget    = [0, -3, groundZ + 2.5];
ax.CameraViewAngle = 20;

ax.Toolbar = [];
fig.Pointer = 'arrow';

% Arrow keys aligned to screen directions: project camera axes onto ground plane
camPos = ax.CameraPosition;
camFwd = -camPos(1:2); camFwd = camFwd / max(norm(camFwd), 1e-10);
camRight = [camFwd(2), -camFwd(1)];

% Mode indicator (top center, over dark background)
modeLabels = {'1: Mouse/Touch: Follow', '2: Arrow Keys: Move', '3: Auto (Lissajous)'};
hModeText = text(ax, 0, 0, 0, '', ...
    'Units','normalized','Position',[0.5 0.98], ...
    'HorizontalAlignment','center','Color',[0.65 0.65 0.45], ...
    'FontSize',11,'FontWeight','bold','Margin',1);

%% ========== CONTROL STATE ==========
ud = struct();
ud.stop       = false;
ud.pause      = false;
ud.mode       = 1;               % 1=mouse, 2=arrows, 3=auto
ud.lightXY    = [0, 0];          % current light XY (shared across modes)
ud.arrowXY    = [0, 0];          % accumulated arrow-key position
ud.mouseTarget = [0, 0];         % raw projected mouse position
ud.mouseSmooth = [0, 0];         % smoothed mouse position
ud.autoBlend   = 1;              % 0→1 transition blend into Lissajous
ud.autoStartXY = [0, 0];         % position when mode 3 was activated
ud.keysHeld   = struct('left',false, 'right',false, 'up',false, 'down',false);
ud.recording   = false;          % R key toggles recording
ud.exportReady = false;          % set true when recording stops → triggers export

% Store constants needed by callbacks
ud.groundZ     = groundZ;
ud.lightHeight = lightHeight;
% Field geometry for trapezoid-aware clamping (mouse/arrow limited to trapezoid + margin)
ud.fieldFrontY = fieldFrontY;
ud.fieldBackY  = fieldBackY;
ud.fieldFrontW = fieldFrontW;
ud.fieldBackW  = fieldBackW;
ud.fieldMargin = 1.5;            % movement padding beyond trapezoid edge

fig.UserData = ud;

% Update mode display
updateModeText(fig, hModeText, modeLabels);

% Wire up callbacks
fig.KeyPressFcn          = @(~,evt) keyDownCB(evt, fig, hModeText, modeLabels);
fig.WindowKeyReleaseFcn  = @(~,evt) keyUpCB(evt, fig);
fig.WindowButtonMotionFcn = @(~,~) mouseCB(fig, ax);

%% ========== ANIMATION LOOP ==========
bloomState = ones(nRoses, 1);   % continuous bloom level per rose
prevLevel  = ones(nRoses, 1);   % previous quantized level (for skip detection)
t = 0;                          % Lissajous time parameter
fixedStep = ischar(referenceFPS) || isstring(referenceFPS);  % 'fixed' mode
lastTic = tic;
frames = {};

while isvalid(fig) && ~fig.UserData.stop
    frameStart = tic;

    % Pause handling
    while isvalid(fig) && ~fig.UserData.stop && fig.UserData.pause
        drawnow; pause(0.05);
        lastTic = tic;  % reset so dt doesn't explode after unpause
    end
    if ~isvalid(fig) || fig.UserData.stop, break; end

    % Frame timing: fixed when recording, real-time otherwise
    if fixedStep || fig.UserData.recording
        frameScale = 1;                              % constant: every frame = 1 tick
    else
        dt = min(toc(lastTic), 0.05);        % real elapsed, capped
        lastTic = tic;
        frameScale = dt * referenceFPS;               % scale to target playback rate
    end

    % --- Compute light XY based on current mode ---
    switch fig.UserData.mode
        case 1  % Mouse follow: smooth interpolation toward cursor
            target = fig.UserData.mouseTarget;
            sm = fig.UserData.mouseSmooth + mouseLerp * frameScale * (target - fig.UserData.mouseSmooth);
            fig.UserData.mouseSmooth = sm;
            lx = sm(1);
            ly = sm(2);

        case 2  % Arrow keys: read held-key state, move aligned to screen
            kh = fig.UserData.keysHeld;
            rawRight = double(kh.right) - double(kh.left);
            rawUp    = double(kh.up)    - double(kh.down);
            vx = arrowSpeed * frameScale * (rawRight * camRight(1) + rawUp * camFwd(1));
            vy = arrowSpeed * frameScale * (rawRight * camRight(2) + rawUp * camFwd(2));
            aPos = fig.UserData.arrowXY + [vx, vy];
            [aPos(1), aPos(2)] = clampToField(aPos(1), aPos(2), fig.UserData);
            fig.UserData.arrowXY = aPos;
            lx = aPos(1);
            ly = aPos(2);

        case 3  % Auto: Lissajous path (follows trapezoidal field shape)
            midY = (fieldFrontY + fieldBackY) / 2;
            halfY = (fieldBackY - fieldFrontY) / 2;
            targetLy = midY + halfY * 0.85 * cos(t * 0.7);
            w = widthAt(targetLy);
            targetLx = w/2 * 0.85 * sin(t * 1.0);

            % Smoothstep blend from start position into Lissajous
            ab = min(1, fig.UserData.autoBlend + 0.8 * frameScale / 30);
            fig.UserData.autoBlend = ab;
            b = ab * ab * (3 - 2 * ab);  % smoothstep
            lx = fig.UserData.autoStartXY(1) + b * (targetLx - fig.UserData.autoStartXY(1));
            ly = fig.UserData.autoStartXY(2) + b * (targetLy - fig.UserData.autoStartXY(2));
    end

    lz = groundZ + lightHeight + 0.3 * sin(t * 0.4);

    % Store current position for seamless mode transitions
    fig.UserData.lightXY = [lx, ly];

    % --- Update light sphere positions ---
    set(hLightSphere,     'XData', lsX + lx, 'YData', lsY + ly, 'ZData', lsZ + lz);
    set(hInfluenceSphere, 'XData', liX + lx, 'YData', liY + ly, 'ZData', liZ + lz);
    hMovingLight.Position = [lx, ly, lz];

    % --- Vectorized distance computation (all roses at once) ---
    dv = rosePos3D - [lx, ly, lz];
    allDists = sqrt(dv(:,1).^2 + dv(:,2).^2 + dv(:,3).^2);

    % --- Vectorized bloom state update ---
    inside    = allDists < influenceRadius;
    rawProx   = max(0, 1 - allDists / influenceRadius);
    proximity = 0.25 + 0.75 * rawProx.^2;   % smooth falloff with floor
    bloomState(inside)  = min(nBloomLevels, bloomState(inside)  + bloomRate * proximity(inside) * frameScale);
    bloomState(~inside) = max(1,             bloomState(~inside) - unbloomRate * frameScale);

    % Quantize to integer bloom levels
    curLevel = max(1, min(nBloomLevels, round(bloomState)));

    % --- Only update roses whose bloom level actually changed ---
    changed = find(curLevel ~= prevLevel);
    for ci = 1:numel(changed)
        ri = changed(ci);
        bi = curLevel(ri);
        pts = [bloomX{bi}(:), bloomY{bi}(:), bloomZ{bi}(:)] * roseScRm{ri};
        set(hRoses(ri), ...
            'XData', reshape(pts(:,1), n, n) + roseOffset(ri,1), ...
            'YData', reshape(pts(:,2), n, n) + roseOffset(ri,2), ...
            'ZData', reshape(pts(:,3), n, n) + roseOffset(ri,3));
    end
    prevLevel = curLevel;

    t = t + lightSpeed * frameScale;
    drawnow;
    if ~isvalid(fig), break; end

    % Capture frame when recording
    if fig.UserData.recording && isvalid(ax)
        frames{end+1} = getframe(ax); %#ok<SAGROW>
    end

    % R key: export triggered by stop
    if fig.UserData.exportReady
        fig.UserData.exportReady = false;
        break;
    end

    % Throttle: if machine is faster than referenceFPS, wait
    if ~fixedStep
        elapsed = toc(frameStart);
        targetDt = 1 / referenceFPS;
        if elapsed < targetDt
            pause(targetDt - elapsed);
        end
    end
end
if isvalid(fig), close(fig); end

%% ========== EXPORT PIPELINE ==========
recordingComplete = ~isempty(frames);

% --- Show export dialog (loops until user cancels) ---
if recordingComplete
    frameData = cellfun(@(f) f.cdata, frames, 'UniformOutput', false);
    fprintf('Recording complete: %d frames captured.\n', numel(frameData));
    while true
        [fmt, fps, dith, skipN] = showExportDialog(numel(frameData));
        if isempty(fmt), break; end

        % Temporal downsampling: keep every Nth frame
        if skipN > 1
            exportData = frameData(1:skipN:end);
        else
            exportData = frameData;
        end

        try
            switch fmt
                case 'mp4'
                    [file, path] = uiputfile('*.mp4', 'Save Video', 'BloomingRoseField.mp4');
                    if file ~= 0
                        exportToVideo(exportData, fullfile(path, file), fps);
                    end
                case 'gif'
                    [file, path] = uiputfile('*.gif', 'Save GIF', 'BloomingRoseField.gif');
                    if file ~= 0
                        exportToGIF(exportData, fullfile(path, file), fps, dith);
                    end
                case 'png'
                    folder = uigetdir(pwd, 'Select Folder for PNG Sequence');
                    if folder ~= 0
                        exportToPNG(exportData, folder);
                    end
            end
        catch ME
            h = errordlg(sprintf('Export failed:\n%s', ME.message), 'Export Error');
            centerDialog(h);
            uiwait(h);
        end
    end
end

%% ========== LOCAL FUNCTIONS ==========

function Rm = buildHeadRotation(tangent, rotZ_angle)
% Build rotation matrix for a rose head from stem-top tangent + Z rotation.
% 1) Rz rotates petals around the local up axis (variety)
% 2) Rt tilts [0,0,1] to align with the stem tangent (Rodrigues' formula)
    cz = cos(rotZ_angle);  sz = sin(rotZ_angle);
    Rz = [cz -sz 0; sz cz 0; 0 0 1];

    T = tangent(:)' / max(norm(tangent), 1e-10);
    v = cross([0 0 1], T);
    sinA = norm(v);
    if sinA > 1e-6
        v = v / sinA;
        cosA = T(3);  % dot([0,0,1], T)
        K = [0 -v(3) v(2); v(3) 0 -v(1); -v(2) v(1) 0];
        Rt = eye(3) + sinA*K + (1-cosA)*(K*K);
    else
        Rt = eye(3);
    end
    Rm = Rt * Rz;
end

function keyDownCB(evt, fig, hModeText, modeLabels)
% Key press: mode switching, movement keys, stop/pause.
    switch evt.Key
        case {'q','x','escape'}
            fig.UserData.stop = true;
        case 'space'
            fig.UserData.pause = ~fig.UserData.pause;
            updateModeText(fig, hModeText, modeLabels);

        % --- Mode switching (seamless handoff) ---
        case '1'
            fig.UserData.mode = 1;
            fig.UserData.mouseSmooth = fig.UserData.lightXY;
            updateModeText(fig, hModeText, modeLabels);
        case '2'
            fig.UserData.mode = 2;
            fig.UserData.arrowXY = fig.UserData.lightXY;
            updateModeText(fig, hModeText, modeLabels);
        case '3'
            fig.UserData.mode = 3;
            fig.UserData.autoStartXY = fig.UserData.lightXY;
            fig.UserData.autoBlend = 0;
            updateModeText(fig, hModeText, modeLabels);

        % --- Arrow key tracking ---
        case 'leftarrow',  fig.UserData.keysHeld.left  = true;
        case 'rightarrow', fig.UserData.keysHeld.right = true;
        case 'uparrow',    fig.UserData.keysHeld.up    = true;
        case 'downarrow',  fig.UserData.keysHeld.down  = true;

        % --- Recording toggle ---
        case 'r'
            if ~fig.UserData.recording
                fig.UserData.recording = true;
                fig.Name = '● REC — Blooming Rose Field';
                hModeText.Visible = 'off';
            else
                fig.UserData.recording = false;
                fig.UserData.exportReady = true;
                fig.Name = 'Blooming Rose Field';
                hModeText.Visible = 'on';
                updateModeText(fig, hModeText, modeLabels);
            end
    end
end

function keyUpCB(evt, fig)
% Key release: clear directional flags for continuous arrow movement.
    switch evt.Key
        case 'leftarrow',  fig.UserData.keysHeld.left  = false;
        case 'rightarrow', fig.UserData.keysHeld.right = false;
        case 'uparrow',    fig.UserData.keysHeld.up    = false;
        case 'downarrow',  fig.UserData.keysHeld.down  = false;
    end
end

function mouseCB(fig, ax)
% WindowButtonMotionFcn: project mouse ray to light-height plane.
% Runs on every mouse move — keep it minimal.
    try %#ok<TRYNC>
        cp  = ax.CurrentPoint;           % [nearPt; farPt] 2×3
        dir = cp(2,:) - cp(1,:);         % ray direction
        targetZ = fig.UserData.groundZ + fig.UserData.lightHeight;
        if abs(dir(3)) > 1e-10
            tHit = (targetZ - cp(1,3)) / dir(3);
            hitX = cp(1,1) + tHit * dir(1);
            hitY = cp(1,2) + tHit * dir(2);
            [hitX, hitY] = clampToField(hitX, hitY, fig.UserData);
            fig.UserData.mouseTarget = [hitX, hitY];
        end
    end
end

function updateModeText(fig, hText, labels)
    if fig.UserData.pause
        pauseLabel = 'Unpause';
    else
        pauseLabel = 'Pause';
    end
    hText.String = [labels{fig.UserData.mode} '   |   1/2/3: Switch   |   Space: ' pauseLabel '   |   R: Record   |   Q: Quit'];
end

function rgb = scalarToRGB(X, Y, Z, cmap)
    C = hypot(hypot(X, Y), Z*0.9);
    cmin = min(C(:));  cmax = max(C(:));
    C_norm = (C - cmin) / (cmax - cmin + eps);
    nC = size(cmap, 1);
    idx = max(1, min(nC, round(C_norm*(nC-1)) + 1));
    rgb = zeros([size(C), 3]);
    rgb(:,:,1) = reshape(cmap(idx(:),1), size(C));
    rgb(:,:,2) = reshape(cmap(idx(:),2), size(C));
    rgb(:,:,3) = reshape(cmap(idx(:),3), size(C));
end

function [cx, cy] = clampToField(x, y, ud)
%CLAMPTOFIELD  Clamp XY to trapezoid + margin.
    m  = ud.fieldMargin;
    cy = max(ud.fieldFrontY - m, min(ud.fieldBackY + m, y));
    % Interpolate trapezoid half-width at clamped Y
    frac = (cy - ud.fieldFrontY) / (ud.fieldBackY - ud.fieldFrontY);
    frac = max(0, min(1, frac));
    halfW = (ud.fieldFrontW + frac * (ud.fieldBackW - ud.fieldFrontW)) / 2 + m;
    cx = max(-halfW, min(halfW, x));
end

%% ========== EXPORT FUNCTIONS ==========

function [fmt, fps, dith, skipN] = showExportDialog(frameCount)
%SHOWEXPORTDIALOG  Modal dialog to choose export format, FPS, frame skip, and dithering.
    fmt   = [];
    fps   = 60;
    dith  = true;
    skipN = 1;

    baseW = 220;  expandedW = 330;  dlgH = 195;

    dlg = uifigure('Name', 'Export Recording', ...
        'Position', [0 0 expandedW dlgH], ...
        'Resize', 'off', 'WindowStyle', 'modal', ...
        'Color', [0.15 0.15 0.15], ...
        'Visible', 'off');

    uilabel(dlg, 'Text', sprintf('Export %d frames as:', frameCount), ...
        'Position', [0 155 baseW 22], ...
        'FontColor', 'w', 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'center');

    fmtDrop = uidropdown(dlg, ...
        'Items', {'MP4 Video', 'Animated GIF', 'PNG Sequence'}, ...
        'ItemsData', {'mp4', 'gif', 'png'}, ...
        'Value', 'mp4', ...
        'Position', [(baseW-160)/2 115 160 26], ...
        'ValueChangedFcn', @(~,~) updateLayout());

    % Frame skip dropdown
    uilabel(dlg, 'Text', 'Keep every:', ...
        'Position', [10 75 65 20], ...
        'FontColor', [0.7 0.7 0.7]);
    skipDrop = uidropdown(dlg, ...
        'Items', {'1 (all)', '2nd', '3rd', '4th', '5th'}, ...
        'ItemsData', {1, 2, 3, 4, 5}, ...
        'Value', 1, ...
        'Position', [80 73 80 26], ...
        'ValueChangedFcn', @(~,~) updateFrameCount());
    skipInfo = uilabel(dlg, 'Text', sprintf('= %d frames', frameCount), ...
        'Position', [165 75 55 20], ...
        'FontColor', [0.5 0.5 0.5], 'FontSize', 10);

    % MP4: clapper icon (right panel)
    fpsIcon = uilabel(dlg, 'Text', char([55356 57260]), ...
        'Position', [225 130 80 45], 'FontSize', 38, ...
        'HorizontalAlignment', 'center');
    fpsLbl = uilabel(dlg, 'Text', 'Video FPS:', ...
        'Position', [225 93 80 20], ...
        'FontColor', [0.7 0.7 0.7], ...
        'HorizontalAlignment', 'center');
    fpsSpin = uispinner(dlg, ...
        'Position', [225 60 80 30], ...
        'Value', 60, 'Limits', [1 240], 'Step', 5, ...
        'ValueDisplayFormat', '%.0f fps');

    % GIF: artist palette icon (right panel)
    gifIcon = uilabel(dlg, 'Text', char([55356 57256]), ...
        'Position', [225 140 80 40], 'FontSize', 34, ...
        'HorizontalAlignment', 'center', 'Visible', 'off');
    dithCheck = uicheckbox(dlg, ...
        'Text', 'Dithering', ...
        'Value', true, ...
        'Position', [235 53 80 22], ...
        'FontColor', [0.7 0.7 0.7], ...
        'Visible', 'off');

    uibutton(dlg, 'push', 'Text', 'Cancel', ...
        'Position', [25 25 70 30], ...
        'ButtonPushedFcn', @(~,~) close(dlg));
    uibutton(dlg, 'push', 'Text', 'Export', ...
        'Position', [115 25 70 30], ...
        'BackgroundColor', [0.2 0.5 0.3], 'FontColor', 'w', ...
        'ButtonPushedFcn', @(~,~) onExport());

    movegui(dlg, 'center');
    dlg.Visible = 'on';
    uiwait(dlg);

    function updateFrameCount()
        n = ceil(frameCount / skipDrop.Value);
        skipInfo.Text = sprintf('= %d frames', n);
    end

    function updateLayout()
        isMp4 = strcmp(fmtDrop.Value, 'mp4');
        isGif = strcmp(fmtDrop.Value, 'gif');
        showExtra = isMp4 || isGif;
        if showExtra
            dlg.Position(3) = expandedW;
        else
            dlg.Position(3) = baseW;
        end
        fpsIcon.Visible   = matlab.lang.OnOffSwitchState(isMp4);
        gifIcon.Visible   = matlab.lang.OnOffSwitchState(isGif);
        fpsLbl.Visible    = matlab.lang.OnOffSwitchState(isMp4 || isGif);
        fpsSpin.Visible   = matlab.lang.OnOffSwitchState(isMp4 || isGif);
        dithCheck.Visible = matlab.lang.OnOffSwitchState(isGif);
        if isGif
            fpsLbl.Text = 'FPS:';
            fpsLbl.Position(2) = 107;
            fpsSpin.Position(2) = 79;
        else
            fpsLbl.Text = 'Video FPS:';
            fpsLbl.Position(2) = 93;
            fpsSpin.Position(2) = 60;
        end
    end

    function onExport()
        fmt   = fmtDrop.Value;
        fps   = fpsSpin.Value;
        dith  = dithCheck.Value;
        skipN = skipDrop.Value;
        close(dlg);
    end
end

function exportToVideo(frameData, filepath, fps)
%EXPORTTOVIDEO  Write frames to an MP4 file.
    try
        nF = numel(frameData);
        wb = waitbar(0, 'Exporting video...');
        v = VideoWriter(filepath, 'MPEG-4');
        v.FrameRate = fps;
        v.Quality   = 95;
        open(v);
        for i = 1:nF
            writeVideo(v, frameData{i});
            if mod(i, 20) == 0 || i == nF
                waitbar(i/nF, wb, sprintf('Exporting video... %d/%d', i, nF));
            end
        end
        close(v);
        close(wb);
        h = msgbox(sprintf('Video saved to:\n%s\n(%d fps)', filepath, fps), ...
            'Export Complete', 'help');
        centerDialog(h);
        uiwait(h);
    catch ME
        if exist('wb', 'var') && isvalid(wb), close(wb); end
        errordlg(ME.message, 'Export Error');
    end
end

function exportToGIF(frameData, filepath, fps, useDither)
%EXPORTTOGIF  Write frames to an animated GIF with optional dithering.
    try
        nF = numel(frameData);
        delayTime = 1 / fps;
        if useDither
            ditherMode = 'dither';
        else
            ditherMode = 'nodither';
        end

        wb = waitbar(0, 'Exporting GIF (building colormap)...');

        % Build global colormap from sampled frames
        sampleIdx = unique(round(linspace(1, nF, min(10, nF))));
        sampledPix = [];
        for idx = sampleIdx
            img = frameData{idx};
            sampled = img(1:4:end, 1:4:end, :);
            sampledPix = [sampledPix; reshape(sampled, [], 3)]; %#ok<AGROW>
        end
        [~, globalCmap] = rgb2ind(reshape(sampledPix, [], 1, 3), 256, ditherMode);

        waitbar(0, wb, 'Exporting GIF...');
        for i = 1:nF
            indexedImg = rgb2ind(frameData{i}, globalCmap, ditherMode);
            if i == 1
                imwrite(indexedImg, globalCmap, filepath, 'gif', ...
                    'LoopCount', Inf, 'DelayTime', delayTime);
            else
                imwrite(indexedImg, globalCmap, filepath, 'gif', ...
                    'WriteMode', 'append', 'DelayTime', delayTime);
            end
            if mod(i, 20) == 0 || i == nF
                waitbar(i/nF, wb, sprintf('Exporting GIF... %d/%d', i, nF));
            end
        end
        close(wb);
        h = msgbox(sprintf('GIF saved to:\n%s', filepath), 'Export Complete', 'help');
        centerDialog(h);
        uiwait(h);
    catch ME
        if exist('wb', 'var') && isvalid(wb), close(wb); end
        errordlg(ME.message, 'Export Error');
    end
end

function exportToPNG(frameData, folderpath)
%EXPORTTOPNG  Write frames as a numbered PNG sequence.
    try
        nF = numel(frameData);
        wb = waitbar(0, 'Exporting PNG sequence...');
        numDigits = max(4, ceil(log10(nF + 1)));
        fmtStr = sprintf('frame_%%0%dd.png', numDigits);
        for i = 1:nF
            imwrite(frameData{i}, fullfile(folderpath, sprintf(fmtStr, i)));
            if mod(i, 20) == 0 || i == nF
                waitbar(i/nF, wb, sprintf('Exporting PNG... %d/%d', i, nF));
            end
        end
        close(wb);
        h = msgbox(sprintf('%d frames saved to:\n%s', nF, folderpath), ...
            'Export Complete', 'help');
        centerDialog(h);
        uiwait(h);
    catch ME
        if exist('wb', 'var') && isvalid(wb), close(wb); end
        errordlg(ME.message, 'Export Error');
    end
end

function centerDialog(h)
%CENTERDIALOG  Move a dialog figure to the center of the current monitor.
    if isvalid(h)
        movegui(h, 'center');
    end
end
