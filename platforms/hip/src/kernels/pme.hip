KERNEL void gridSpreadCharge(GLOBAL const real4* RESTRICT posq,
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL mm_ulong* RESTRICT pmeGrid,
#else
        GLOBAL real* RESTRICT pmeGrid,
#endif
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const int2* RESTRICT pmeAtomGridIndex,
#ifdef CHARGE_FROM_SIGEPS
        GLOBAL const float2* RESTRICT sigmaEpsilon
#else
        GLOBAL const real* RESTRICT charges
#endif
        ) {
    // Process the atoms in spatially sorted order.  This improves efficiency when writing
    // the grid values.  PME_ORDER lanes process one atom.

    real3 data[PME_ORDER];
    const real scale = RECIP((real) (PME_ORDER-1));
    const int lanesPerAtom = PME_ORDER;
    const int atomsPerWarp = warpSize / lanesPerAtom;
    const int atomsPerBlock = atomsPerWarp * (LOCAL_SIZE / warpSize);
    const int warpInBlock = LOCAL_ID / warpSize;
    const int laneId = LOCAL_ID % warpSize;
    const int atomInWarp = laneId / lanesPerAtom;
    const int atomStartLane = atomInWarp * lanesPerAtom;
    const int laneInAtom = laneId%lanesPerAtom;
    if (atomInWarp >= atomsPerWarp)
        return;
    for (int atomi = GROUP_ID*atomsPerBlock + warpInBlock*atomsPerWarp + atomInWarp; atomi < NUM_ATOMS; atomi += NUM_GROUPS*atomsPerBlock) {
        int atom = pmeAtomGridIndex[atomi].x;
        real4 pos = posq[atom];
#ifdef CHARGE_FROM_SIGEPS
        const float2 sigEps = sigmaEpsilon[atom];
        const real charge = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
        const real charge = (CHARGE)*EPSILON_FACTOR;
#endif
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);
        if (charge == 0)
            continue;

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.
        // One lane calculates thetas for one dimension (lane 0 - x, 1 - y, 2 - z), then all lanes
        // of the atom read these values using __shfl.

        if (laneInAtom < 3) {
            real t0 = (laneInAtom == 0 ? t.x : (laneInAtom == 1 ? t.y : t.z));
            real dr = t0-(int) t0;
            data[PME_ORDER-1].x = 0;
            data[1].x = dr;
            data[0].x = 1-dr;
            for (int j = 3; j < PME_ORDER; j++) {
                real div = RECIP((real) (j-1));
                data[j-1].x = div*dr*data[j-2].x;
                for (int k = 1; k < (j-1); k++)
                    data[j-k-1].x = div*((dr+k)*data[j-k-2].x + (j-k-dr)*data[j-k-1].x);
                data[0].x = div*(1-dr)*data[0].x;
            }
            data[PME_ORDER-1].x = scale*dr*data[PME_ORDER-2].x;
            for (int j = 1; j < (PME_ORDER-1); j++)
                data[PME_ORDER-j-1].x = scale*((dr+j)*data[PME_ORDER-j-2].x + (PME_ORDER-j-dr)*data[PME_ORDER-j-1].x);
            data[0].x = scale*(1-dr)*data[0].x;
        }
        for (int i = 0; i < PME_ORDER; i++) {
            data[i] = make_real3(__shfl(data[i].x, atomStartLane + 0),
                                 __shfl(data[i].x, atomStartLane + 1),
                                 __shfl(data[i].x, atomStartLane + 2));
        }

        // Spread the charge from this atom onto each grid point.  PME_ORDER lanes access
        // consecutive addresses.

        int iz = laneInAtom;
        int zindex = gridIndex.z+iz;
        zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
        real dz = 0;
        for (int i = 0; i < PME_ORDER; i++) {
            dz = i == iz ? data[i].z : dz;
        }
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dx = charge*data[ix].x;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real dxdy = dx*data[iy].y;
                int index = ybase + zindex;
                real add = dxdy*dz;
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
                ATOMIC_ADD(&pmeGrid[index], (mm_ulong) realToFixedPoint(add));
#else
                ATOMIC_ADD(&pmeGrid[index], add);
#endif
            }
        }
    }
}

KERNEL void finishSpreadCharge(
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        GLOBAL const mm_long* RESTRICT grid1,
#else
        GLOBAL const real* RESTRICT grid1,
#endif
        GLOBAL real* RESTRICT grid2) {
    const unsigned int gridSize = GRID_SIZE_X*GRID_SIZE_Y*GRID_SIZE_Z;
    real scale = 1/(real) 0x100000000;
    for (int index = GLOBAL_ID; index < gridSize; index += GLOBAL_SIZE) {
#ifdef USE_FIXED_POINT_CHARGE_SPREADING
        grid2[index] = scale*grid1[index];
#else
        grid2[index] = grid1[index];
#endif
    }
}

KERNEL void gridInterpolateForce(GLOBAL const real4* RESTRICT posq, GLOBAL mm_ulong* RESTRICT forceBuffers, GLOBAL const real* RESTRICT pmeGrid,
        real4 periodicBoxSize, real4 invPeriodicBoxSize, real4 periodicBoxVecX, real4 periodicBoxVecY, real4 periodicBoxVecZ,
        real4 recipBoxVecX, real4 recipBoxVecY, real4 recipBoxVecZ, GLOBAL const int2* RESTRICT pmeAtomGridIndex,
#ifdef CHARGE_FROM_SIGEPS
        GLOBAL const float2* RESTRICT sigmaEpsilon
#else
        GLOBAL const real* RESTRICT charges
#endif
        ) {
    real3 data[PME_ORDER];
    real3 ddata[PME_ORDER];
    const real scale = RECIP((real) (PME_ORDER-1));
    
    // Process the atoms in spatially sorted order.  This improves cache performance when loading
    // the grid values.  PME_ORDER lanes process one atom.

    const int lanesPerAtom = PME_ORDER;
    const int atomsPerWarp = warpSize / lanesPerAtom;
    const int atomsPerBlock = atomsPerWarp * (LOCAL_SIZE / warpSize);
    const int warpInBlock = LOCAL_ID / warpSize;
    const int laneId = LOCAL_ID % warpSize;
    const int atomInWarp = laneId / lanesPerAtom;
    const int atomStartLane = atomInWarp * lanesPerAtom;
    const int laneInAtom = laneId%lanesPerAtom;
    if (atomInWarp >= atomsPerWarp)
        return;
    for (int atomi = GROUP_ID*atomsPerBlock + warpInBlock*atomsPerWarp + atomInWarp; atomi < NUM_ATOMS; atomi += NUM_GROUPS*atomsPerBlock) {
        int atom = pmeAtomGridIndex[atomi].x;
        real3 force = make_real3(0);
        real4 pos = posq[atom];
        APPLY_PERIODIC_TO_POS(pos)
        real3 t = make_real3(pos.x*recipBoxVecX.x+pos.y*recipBoxVecY.x+pos.z*recipBoxVecZ.x,
                             pos.y*recipBoxVecY.y+pos.z*recipBoxVecZ.y,
                             pos.z*recipBoxVecZ.z);
        t.x = (t.x-floor(t.x))*GRID_SIZE_X;
        t.y = (t.y-floor(t.y))*GRID_SIZE_Y;
        t.z = (t.z-floor(t.z))*GRID_SIZE_Z;
        int3 gridIndex = make_int3(((int) t.x) % GRID_SIZE_X,
                                   ((int) t.y) % GRID_SIZE_Y,
                                   ((int) t.z) % GRID_SIZE_Z);

        // Since we need the full set of thetas, it's faster to compute them here than load them
        // from global memory.
        // One lane calculates thetas for one dimension (lane 0 - x, 1 - y, 2 - z), then all lanes
        // of the atom read these values using __shfl.

        if (laneInAtom < 3) {
            real t0 = (laneInAtom == 0 ? t.x : (laneInAtom == 1 ? t.y : t.z));
            real dr = t0-(int) t0;
            data[PME_ORDER-1].x = 0;
            data[1].x = dr;
            data[0].x = 1-dr;
            for (int j = 3; j < PME_ORDER; j++) {
                real div = RECIP((real) (j-1));
                data[j-1].x = div*dr*data[j-2].x;
                for (int k = 1; k < (j-1); k++)
                    data[j-k-1].x = div*((dr+k)*data[j-k-2].x + (j-k-dr)*data[j-k-1].x);
                data[0].x = div*(1-dr)*data[0].x;
            }
            ddata[0].x = -data[0].x;
            for (int j = 1; j < PME_ORDER; j++)
                ddata[j].x = data[j-1].x-data[j].x;
            data[PME_ORDER-1].x = scale*dr*data[PME_ORDER-2].x;
            for (int j = 1; j < (PME_ORDER-1); j++)
                data[PME_ORDER-j-1].x = scale*((dr+j)*data[PME_ORDER-j-2].x + (PME_ORDER-j-dr)*data[PME_ORDER-j-1].x);
            data[0].x = scale*(1-dr)*data[0].x;
        }
        for (int i = 0; i < PME_ORDER; i++) {
            data[i] = make_real3(__shfl(data[i].x, atomStartLane + 0),
                                 __shfl(data[i].x, atomStartLane + 1),
                                 __shfl(data[i].x, atomStartLane + 2));
            ddata[i] = make_real3(__shfl(ddata[i].x, atomStartLane + 0),
                                  __shfl(ddata[i].x, atomStartLane + 1),
                                  __shfl(ddata[i].x, atomStartLane + 2));
        }

        // Compute the force on this atom.  PME_ORDER lanes access consecutive addresses.

        int iz = laneInAtom;
        int zindex = gridIndex.z+iz;
        zindex -= (zindex >= GRID_SIZE_Z ? GRID_SIZE_Z : 0);
        real dz = 0;
        real ddz = 0;
        for (int i = 0; i < PME_ORDER; i++) {
            dz = i == iz ? data[i].z : dz;
            ddz = i == iz ? ddata[i].z : ddz;
        }
        for (int ix = 0; ix < PME_ORDER; ix++) {
            int xbase = gridIndex.x+ix;
            xbase -= (xbase >= GRID_SIZE_X ? GRID_SIZE_X : 0);
            xbase = xbase*GRID_SIZE_Y*GRID_SIZE_Z;
            real dx = data[ix].x;
            real ddx = ddata[ix].x;
            for (int iy = 0; iy < PME_ORDER; iy++) {
                int ybase = gridIndex.y+iy;
                ybase -= (ybase >= GRID_SIZE_Y ? GRID_SIZE_Y : 0);
                ybase = xbase + ybase*GRID_SIZE_Z;
                real dy = data[iy].y;
                real ddy = ddata[iy].y;
                int index = ybase + zindex;
                real gridvalue = pmeGrid[index];
                force.x += ddx*dy*dz*gridvalue;
                force.y += dx*ddy*dz*gridvalue;
                force.z += dx*dy*ddz*gridvalue;
            }
        }

        // Sum forces of all lanes of the current atom and write each component by one lane.

        real3 f = force;
        force = make_real3(0);
        for (int i = 0; i < PME_ORDER; i++) {
            force += make_real3(__shfl(f.x, atomStartLane + i),
                                __shfl(f.y, atomStartLane + i),
                                __shfl(f.z, atomStartLane + i));
        }
#ifdef CHARGE_FROM_SIGEPS
        const float2 sigEps = sigmaEpsilon[atom];
        real q = 8*sigEps.x*sigEps.x*sigEps.x*sigEps.y;
#else
        real q = CHARGE*EPSILON_FACTOR;
#endif
        real forceX = -q*(force.x*GRID_SIZE_X*recipBoxVecX.x);
        real forceY = -q*(force.x*GRID_SIZE_X*recipBoxVecY.x+force.y*GRID_SIZE_Y*recipBoxVecY.y);
        real forceZ = -q*(force.x*GRID_SIZE_X*recipBoxVecZ.x+force.y*GRID_SIZE_Y*recipBoxVecZ.y+force.z*GRID_SIZE_Z*recipBoxVecZ.z);
        if (laneInAtom < 3) {
            real f = (laneInAtom == 0 ? forceX : (laneInAtom == 1 ? forceY : forceZ));
#ifdef USE_PME_STREAM
            ATOMIC_ADD(&forceBuffers[atom+laneInAtom*PADDED_NUM_ATOMS], (mm_ulong) realToFixedPoint(f));
#else
            forceBuffers[atom+laneInAtom*PADDED_NUM_ATOMS] += (mm_ulong) realToFixedPoint(f);
#endif
        }
    }
}
