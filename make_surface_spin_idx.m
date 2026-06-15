function spin_idx = make_surface_spin_idx(coords_lh, coords_rh, nPerm, method)
% MAKE_SURFACE_SPIN_IDX
%
% Generate spin-permutation indices for cortical parcels projected onto
% spherical surfaces.
%
% Repository accompanying:
% Shikauchi et al. (2026)
% https://doi.org/10.64898/2026.02.27.708484
%
% This function implements surface-based spin permutations following
% Alexander-Bloch et al. (2018) and Vázquez-Rodríguez et al. (2019).
% Parcel centroids are rotated on the spherical surface while preserving
% their relative spatial configuration within each hemisphere. The rotated
% parcels are then reassigned to original parcel locations using either
% nearest-neighbor matching ("VR") or one-to-one Hungarian matching
% ("Hungarian").
%
% INPUTS
%   coords_lh : [nL × 3]
%       Spherical coordinates of left-hemisphere parcel centroids.
%
%   coords_rh : [nR × 3]
%       Spherical coordinates of right-hemisphere parcel centroids.
%
%   nPerm : scalar
%       Number of spin permutations.
%
%   method : string
%       Parcel reassignment method:
%           "VR"        - nearest-neighbor assignment
%                         (Vázquez-Rodríguez et al., 2019)
%           "Hungarian" - one-to-one assignment using the Hungarian
%                         algorithm
%
% OUTPUT
%   spin_idx : [(nL+nR) × nPerm]
%       Permutation indices. Each column contains the parcel mapping for
%       one spin permutation and can be used as:
%
%           rotated_map = original_map(spin_idx(:,p));
%
% REFERENCES
%   Alexander-Bloch et al. (2018)
%   Vázquez-Rodríguez et al. (2019)

nL = size(coords_lh, 1);
nR = size(coords_rh, 1);

spin_idx_lh = nan(nL, nPerm);
spin_idx_rh = nan(nR, nPerm);

% Reflection matrix used to generate symmetric rotations across hemispheres
reflect = diag([-1 1 1]);  

for p = 1:nPerm

    % Generate a random rotation matrix
    [Q, ~] = qr(randn(3));

    % Ensure a proper rotation (determinant = +1)
    if det(Q) < 0
        Q(:,1) = -Q(:,1);
    end

    % Apply matched rotations to left and right hemispheres
    R_lh = Q;
    R_rh = reflect * Q * reflect;

    % Rotate parcel centroids on the sphere
    coords_lh_rot = coords_lh * R_lh;
    coords_rh_rot = coords_rh * R_rh;

    if strcmp(method,"VR")
        % Reassign each rotated parcel to its nearest original parcel
        idx_lh = knnsearch(coords_lh, coords_lh_rot);
        idx_rh = knnsearch(coords_rh, coords_rh_rot);
    elseif strcmp(method,"Hungarian") 
        % One-to-one assignment using the Hungarian algorithm
        D_lh = pdist2(coords_lh_rot, coords_lh);
        pairs_lh = matchpairs(D_lh, 1e9);
        idx_lh = nan(nL,1);
        idx_lh(pairs_lh(:,1)) = pairs_lh(:,2);

        D_rh = pdist2(coords_rh_rot, coords_rh);
        pairs_rh = matchpairs(D_rh, 1e9);
        idx_rh = nan(nR,1);
        idx_rh(pairs_rh(:,1)) = pairs_rh(:,2);
    end

    spin_idx_lh(:, p) = idx_lh;
    spin_idx_rh(:, p) = idx_rh;
end

% Combine hemispheres into a single index matrix
spin_idx = nan(nL + nR, nPerm);
spin_idx(1:nL, :) = spin_idx_lh;
spin_idx(nL+1:nL+nR, :) = spin_idx_rh + nL;

end
