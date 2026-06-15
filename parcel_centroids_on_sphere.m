function coords = parcel_centroids_on_sphere(sphere_xyz, labels, parcel_ids)
% PARCEL_CENTROIDS_ON_SPHERE
%
% Compute spherical centroid coordinates for cortical parcels.
%
% Repository accompanying:
% Shikauchi et al. (2026)
% https://doi.org/10.64898/2026.02.27.708484
%
% This function calculates parcel centroids directly on a spherical
% surface mesh. For each parcel, vertex coordinates are averaged and
% subsequently projected back onto the unit sphere. The resulting
% coordinates can be used for surface-based spin permutation analyses.
%
% INPUTS
%   sphere_xyz : [nVertices × 3]
%       Cartesian coordinates of vertices on the spherical surface.
%
%   labels : [nVertices × 1]
%       Parcel label assigned to each vertex.
%
%   parcel_ids : [nParcel × 1]
%       Parcel identifiers for which centroids should be computed.
%
% OUTPUT
%   coords : [nParcel × 3]
%       Unit-length centroid coordinates for each parcel.
%
% EXAMPLE
%   coords = parcel_centroids_on_sphere( ...
%       sphere_xyz, labels, parcel_ids);
%
% NOTE
%   Centroids are computed as the mean Cartesian coordinate of all
%   vertices belonging to a parcel and are subsequently normalized to
%   lie on the unit sphere.

nParcel = length(parcel_ids);
coords = nan(nParcel, 3);

for i = 1:nParcel
    idx = labels == parcel_ids(i);

    if ~any(idx)
        warning('Parcel %d has no vertices.', parcel_ids(i));
        continue
    end

    % Mean Cartesian coordinate of vertices within the parcel
    c = mean(sphere_xyz(idx, :), 1);

    % Project centroid back onto the unit sphere
    coords(i, :) = c ./ norm(c);
end

end