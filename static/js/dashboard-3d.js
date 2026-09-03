import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

/* ============================================================
   NER LANDSLIDE EWS — 3D HAZARD TERRAIN + FLOW SIMULATION
   ------------------------------------------------------------
   Reuses the exact same backend endpoint as the 2D dashboard
   (/api/predict/grid) and the same live data sources (Open-
   Elevation for real terrain, Open-Meteo/ground sensors for rainfall/soil
   moisture, plus USGS seismic activity), just at a denser grid resolution and rendered as
   a rotatable 3D surface instead of a 2D heat layer.

   The "flow simulation" layer is a simplified downslope
   steepest-descent walk from each high-hazard grid cell — it is
   an illustrative visualization, not a calibrated runout model
   (the trained model only outputs point hazard probability, not
   spread dynamics). This is stated in the UI, not just here.
   ============================================================ */

/* ---------------- CONFIG (mirrors dashboard.html's constants) ---------------- */
const GRID_BOUNDS = { latMin: 22.0, latMax: 29.5, lonMin: 88.0, lonMax: 97.5 };
const GRID_N = 21;
const RENDER_N = 161;
const ELEV_CHUNK_SIZE = 150;
const ELEV_BASE_SCALE = 0.02;

const DISTRICTS = [
  { id: 'kamrup_metro', name: 'Kamrup Metropolitan', state: 'Assam', lat: 26.144, lon: 91.736 },
  { id: 'dima_hasao', name: 'Dima Hasao', state: 'Assam', lat: 25.170, lon: 93.017 },
  { id: 'karbi_anglong', name: 'Karbi Anglong', state: 'Assam', lat: 25.840, lon: 93.440 },
  { id: 'cachar', name: 'Cachar', state: 'Assam', lat: 24.833, lon: 92.779 },
  { id: 'sonitpur', name: 'Sonitpur', state: 'Assam', lat: 26.634, lon: 92.800 },
  { id: 'east_khasi_hills', name: 'East Khasi Hills', state: 'Meghalaya', lat: 25.578, lon: 91.893 },
  { id: 'west_khasi_hills', name: 'West Khasi Hills', state: 'Meghalaya', lat: 25.517, lon: 91.267 },
  { id: 'ri_bhoi', name: 'Ri Bhoi', state: 'Meghalaya', lat: 25.900, lon: 91.883 },
  { id: 'west_garo_hills', name: 'West Garo Hills', state: 'Meghalaya', lat: 25.514, lon: 90.220 },
  { id: 'east_jaintia_hills', name: 'East Jaintia Hills', state: 'Meghalaya', lat: 25.350, lon: 92.350 },
  { id: 'papum_pare', name: 'Papum Pare', state: 'Arunachal Pradesh', lat: 27.100, lon: 93.620 },
  { id: 'west_kameng', name: 'West Kameng', state: 'Arunachal Pradesh', lat: 27.264, lon: 92.405 },
  { id: 'upper_subansiri', name: 'Upper Subansiri', state: 'Arunachal Pradesh', lat: 27.983, lon: 94.050 },
  { id: 'lower_dibang_valley', name: 'Lower Dibang Valley', state: 'Arunachal Pradesh', lat: 28.140, lon: 95.840 },
  { id: 'tawang', name: 'Tawang', state: 'Arunachal Pradesh', lat: 27.586, lon: 91.868 },
  { id: 'kohima', name: 'Kohima', state: 'Nagaland', lat: 25.674, lon: 94.108 },
  { id: 'dimapur', name: 'Dimapur', state: 'Nagaland', lat: 25.908, lon: 93.727 },
  { id: 'mokokchung', name: 'Mokokchung', state: 'Nagaland', lat: 26.325, lon: 94.514 },
  { id: 'phek', name: 'Phek', state: 'Nagaland', lat: 25.667, lon: 94.467 },
  { id: 'imphal_west', name: 'Imphal West', state: 'Manipur', lat: 24.817, lon: 93.937 },
  { id: 'churachandpur', name: 'Churachandpur', state: 'Manipur', lat: 24.333, lon: 93.683 },
  { id: 'ukhrul', name: 'Ukhrul', state: 'Manipur', lat: 25.100, lon: 94.367 },
  { id: 'senapati', name: 'Senapati', state: 'Manipur', lat: 25.267, lon: 94.017 },
  { id: 'aizawl', name: 'Aizawl', state: 'Mizoram', lat: 23.727, lon: 92.717 },
  { id: 'lunglei', name: 'Lunglei', state: 'Mizoram', lat: 22.883, lon: 92.733 },
  { id: 'champhai', name: 'Champhai', state: 'Mizoram', lat: 23.470, lon: 93.330 },
  { id: 'kolasib', name: 'Kolasib', state: 'Mizoram', lat: 24.223, lon: 92.676 },
  { id: 'west_tripura', name: 'West Tripura', state: 'Tripura', lat: 23.836, lon: 91.279 },
  { id: 'dhalai', name: 'Dhalai', state: 'Tripura', lat: 23.933, lon: 91.850 },
  { id: 'gomati', name: 'Gomati', state: 'Tripura', lat: 23.533, lon: 91.483 },
  { id: 'east_sikkim', name: 'East Sikkim', state: 'Sikkim', lat: 27.330, lon: 88.610 },
  { id: 'north_sikkim', name: 'North Sikkim', state: 'Sikkim', lat: 27.517, lon: 88.533 },
  { id: 'south_sikkim', name: 'South Sikkim', state: 'Sikkim', lat: 27.167, lon: 88.367 },
  { id: 'west_sikkim', name: 'West Sikkim', state: 'Sikkim', lat: 27.283, lon: 88.267 },
];

const meanLatRad = ((GRID_BOUNDS.latMin + GRID_BOUNDS.latMax) / 2) * Math.PI / 180;
const UNIT_PER_DEG_LAT = 44;
const UNIT_PER_DEG_LON = UNIT_PER_DEG_LAT * Math.cos(meanLatRad);
const WORLD_WIDTH = (GRID_BOUNDS.lonMax - GRID_BOUNDS.lonMin) * UNIT_PER_DEG_LON;
const WORLD_DEPTH = (GRID_BOUNDS.latMax - GRID_BOUNDS.latMin) * UNIT_PER_DEG_LAT;
const KM_PER_DEG_LAT = 111.32;
const KM_PER_DEG_LON = KM_PER_DEG_LAT * Math.cos(meanLatRad);
const RENDER_CELL_METERS_X =
  (GRID_BOUNDS.lonMax - GRID_BOUNDS.lonMin) /
  (RENDER_N - 1) *
  KM_PER_DEG_LON *
  1000;

const RENDER_CELL_METERS_Y =
  (GRID_BOUNDS.latMax - GRID_BOUNDS.latMin) /
  (RENDER_N - 1) *
  KM_PER_DEG_LAT *
  1000;

/* ---------------- REAL BASEMAP IMAGERY ---------------- */
const TILE_TARGET_ACROSS = 8;

const TILE_ZOOM = Math.max(
  4,
  Math.min(
    11,
    Math.round(
      Math.log2(
        360 /
        (GRID_BOUNDS.lonMax - GRID_BOUNDS.lonMin) *
        TILE_TARGET_ACROSS
      )
    )
  )
);

function lonToTileXFloat(lon, z) {
  return (
    (lon + 180) /
    360 *
    Math.pow(2, z)
  );
}

function latToTileYFloat(lat, z) {

  const latRad =
    lat * Math.PI / 180;

  return (
    1 -
    Math.log(
      Math.tan(latRad) +
      1 / Math.cos(latRad)
    ) /
    Math.PI
  ) /
  2 *
  Math.pow(2, z);
}

const TILE_X_MIN =
  Math.floor(
    lonToTileXFloat(
      GRID_BOUNDS.lonMin,
      TILE_ZOOM
    )
  );

const TILE_X_MAX =
  Math.floor(
    lonToTileXFloat(
      GRID_BOUNDS.lonMax,
      TILE_ZOOM
    )
  );

const TILE_Y_MIN =
  Math.floor(
    latToTileYFloat(
      GRID_BOUNDS.latMax,
      TILE_ZOOM
    )
  );

const TILE_Y_MAX =
  Math.floor(
    latToTileYFloat(
      GRID_BOUNDS.latMin,
      TILE_ZOOM
    )
  );

const TILE_SPAN_X =
  TILE_X_MAX -
  TILE_X_MIN +
  1;

const TILE_SPAN_Y =
  TILE_Y_MAX -
  TILE_Y_MIN +
  1;

const TILE_PX = 256;

function basemapUV(lat, lon) {

  const u =
    (
      lonToTileXFloat(
        lon,
        TILE_ZOOM
      ) -
      TILE_X_MIN
    ) /
    TILE_SPAN_X;

  const v =
    (
      latToTileYFloat(
        lat,
        TILE_ZOOM
      ) -
      TILE_Y_MIN
    ) /
    TILE_SPAN_Y;

  return {
    u: Math.max(
      0,
      Math.min(
        1,
        u
      )
    ),

    v: Math.max(
      0,
      Math.min(
        1,
        v
      )
    )
  };
}

/* ---------------- SMALL HELPERS ---------------- */

function latLonToXZ(
  lat,
  lon
) {

  const x =
    (
      lon -
      GRID_BOUNDS.lonMin
    ) /
    (
      GRID_BOUNDS.lonMax -
      GRID_BOUNDS.lonMin
    ) *
    WORLD_WIDTH -
    WORLD_WIDTH / 2;

  const z =
    (
      GRID_BOUNDS.latMax -
      lat
    ) /
    (
      GRID_BOUNDS.latMax -
      GRID_BOUNDS.latMin
    ) *
    WORLD_DEPTH -
    WORLD_DEPTH / 2;

  return {
    x,
    z
  };
}


function haversineKm(
  lat1,
  lon1,
  lat2,
  lon2
) {

  const R = 6371;

  const dLat =
    (
      lat2 -
      lat1
    ) *
    Math.PI /
    180;

  const dLon =
    (
      lon2 -
      lon1
    ) *
    Math.PI /
    180;

  const a =
    Math.sin(
      dLat / 2
    ) ** 2 +

    Math.cos(
      lat1 * Math.PI / 180
    ) *

    Math.cos(
      lat2 * Math.PI / 180
    ) *

    Math.sin(
      dLon / 2
    ) ** 2;

  return (
    R *
    2 *
    Math.atan2(
      Math.sqrt(a),
      Math.sqrt(
        1 -
        a
      )
    )
  );
}


function riskColorCss(
  pct
) {

  if (
    pct >= 70
  ) {

    return '#ef4444';

  }


  if (
    pct >= 40
  ) {

    return '#f59e0b';

  }


  return '#10b981';

}


function lerpColor(
  hexA,
  hexB,
  t
) {

  const a =
    new THREE.Color(
      hexA
    );

  const b =
    new THREE.Color(
      hexB
    );

  return a
    .clone()
    .lerp(
      b,
      Math.max(
        0,
        Math.min(
          1,
          t
        )
      )
    );

}


/* ---------------- TERRAIN ALBEDO ---------------- */

const _cLow =
  new THREE.Color(
    0x3a3f38
  );

const _cMid =
  new THREE.Color(
    0x8d9186
  );

const _cHigh =
  new THREE.Color(
    0xf2f4f7
  );

const _cRock =
  new THREE.Color(
    0x585b53
  );


function terrainAlbedo(
  normElev,
  slopeDeg
) {

  let base;

  if (
    normElev < 0.5
  ) {

    base =
      _cLow
        .clone()
        .lerp(
          _cMid,
          normElev * 2
        );

  } else {

    base =
      _cMid
        .clone()
        .lerp(
          _cHigh,
          (
            normElev -
            0.5
          ) * 2
        );

  }

  const rockT =
    Math.max(
      0,
      Math.min(
        1,
        (
          slopeDeg -
          28
        ) / 30
      )
    );

  if (
    rockT > 0
  ) {

    base.lerp(
      _cRock,
      rockT * 0.7
    );

  }

  return base;

}


/* ---------------- VALUE NOISE ---------------- */

function hash2(
  x,
  y
) {

  const s =
    Math.sin(
      x * 127.1 +
      y * 311.7
    ) *
    43758.5453123;

  return (
    s -
    Math.floor(s)
  );

}


function valueNoise2(
  x,
  y
) {

  const x0 =
    Math.floor(x);

  const y0 =
    Math.floor(y);

  const tx =
    x -
    x0;

  const ty =
    y -
    y0;

  const fade =
    t =>
      t *
      t *
      (3 - 2 * t);

  const a =
    hash2(
      x0,
      y0
    );

  const b =
    hash2(
      x0 + 1,
      y0
    );

  const c =
    hash2(
      x0,
      y0 + 1
    );

  const d =
    hash2(
      x0 + 1,
      y0 + 1
    );

  const u =
    fade(tx);

  const v =
    fade(ty);

  return (
    (
      a * (1 - u) +
      b * u
    ) *
    (1 - v) +

    (
      c * (1 - u) +
      d * u
    ) *
    v
  );

}


function fbmNoise2(
  x,
  y
) {

  let sum = 0;
  let amp = 0.6;
  let freq = 1;
  let norm = 0;

  for (
    let o = 0;
    o < 3;
    o++
  ) {

    sum +=
      (
        valueNoise2(
          x * freq,
          y * freq
        ) -
        0.5
      ) *
      amp;

    norm += amp;

    amp *= 0.5;

    freq *= 2.3;

  }

  return (
    sum /
    norm
  );

}


/* ---------------- BICUBIC SAMPLING ---------------- */

function catmullRom1D(
  p0,
  p1,
  p2,
  p3,
  t
) {

  return (
    0.5 *
    (
      (
        2 *
        p1
      ) +

      (
        -p0 +
        p2
      ) *
      t +

      (
        2 * p0 -
        5 * p1 +
        4 * p2 -
        p3
      ) *
      t *
      t +

      (
        -p0 +
        3 * p1 -
        3 * p2 +
        p3
      ) *
      t *
      t *
      t
    )
  );

}


function coarseElevAt(
  ix,
  iy
) {

  const cx =
    Math.max(
      0,
      Math.min(
        GRID_N - 1,
        ix
      )
    );

  const cy =
    Math.max(
      0,
      Math.min(
        GRID_N - 1,
        iy
      )
    );

  return (
    state
      .grid[
        cy * GRID_N + cx
      ]
      .elevation
  );

}


function coarseProbAt(
  ix,
  iy
) {

  const cx =
    Math.max(
      0,
      Math.min(
        GRID_N - 1,
        ix
      )
    );

  const cy =
    Math.max(
      0,
      Math.min(
        GRID_N - 1,
        iy
      )
    );

  return (
    state
      .grid[
        cy * GRID_N + cx
      ]
      .probability ||
    0
  );

}


function bicubicSample(
  fx,
  fy,
  sampleFn
) {

  const cx =
    Math.max(
      0,
      Math.min(
        GRID_N - 1.0001,
        fx
      )
    );

  const cy =
    Math.max(
      0,
      Math.min(
        GRID_N - 1.0001,
        fy
      )
    );

  const ix0 =
    Math.floor(cx);

  const iy0 =
    Math.floor(cy);

  const tx =
    cx -
    ix0;

  const ty =
    cy -
    iy0;

  const rows = [];

  for (
    let j = -1;
    j <= 2;
    j++
  ) {

    const p0 =
      sampleFn(
        ix0 - 1,
        iy0 + j
      );

    const p1 =
      sampleFn(
        ix0,
        iy0 + j
      );

    const p2 =
      sampleFn(
        ix0 + 1,
        iy0 + j
      );

    const p3 =
      sampleFn(
        ix0 + 2,
        iy0 + j
      );

    rows.push(
      catmullRom1D(
        p0,
        p1,
        p2,
        p3,
        tx
      )
    );

  }

  return catmullRom1D(
    rows[0],
    rows[1],
    rows[2],
    rows[3],
    ty
  );

}


function bilinearElevation(
  fx,
  fy
) {

  return bicubicSample(
    fx,
    fy,
    coarseElevAt
  );

}


function bicubicProbability(
  fx,
  fy
) {

  return Math.max(
    0,
    Math.min(
      1,
      bicubicSample(
        fx,
        fy,
        coarseProbAt
      )
    )
  );

}


function districtName(
  id
) {

  const d =
    DISTRICTS.find(
      x => x.id === id
    );

  return d
    ? `${d.name}, ${d.state}`
    : id;

}


function nearestDistrictId(
  lat,
  lon
) {

  let best =
    DISTRICTS[0];

  let bestDist =
    Infinity;

  DISTRICTS.forEach(
    d => {

      const dist =
        haversineKm(
          lat,
          lon,
          d.lat,
          d.lon
        );

      if (
        dist <
        bestDist
      ) {

        bestDist =
          dist;

        best =
          d;

      }

    }
  );

  return best.id;

}


/* ---------------- TOAST ---------------- */

function toast(
  message,
  type = 'info',
  ms = 4200
) {

  const colors = {

    info:
      'border-cyan-600 text-cyan-300',

    danger:
      'border-red-600 text-red-300',

    success:
      'border-emerald-600 text-emerald-300',

    warn:
      'border-amber-600 text-amber-300'

  };


  const icons = {

    info:
      'fa-circle-info',

    danger:
      'fa-triangle-exclamation',

    success:
      'fa-circle-check',

    warn:
      'fa-bell'

  };


  const el =
    document.createElement(
      'div'
    );


  el.className =
    `bg-panel border ${
      colors[type] ||
      colors.info
    } rounded-lg px-3 py-2.5 text-xs shadow-lg flex items-start gap-2 fade-in`;


  el.innerHTML =
    `<i class="fa-solid ${
      icons[type] ||
      icons.info
    } mt-0.5"></i><span>${message}</span>`;


  document
    .getElementById(
      'toastContainer'
    )
    .appendChild(el);


  setTimeout(
    () => {

      el.style.transition =
        'opacity .3s';

      el.style.opacity =
        '0';

      setTimeout(
        () =>
          el.remove(),
        300
      );

    },
    ms
  );

}


/* ============================================================
   DATA PIPELINE
   ============================================================ */

function buildGridPoints() {

  const points = [];

  for (
    let iy = 0;
    iy < GRID_N;
    iy++
  ) {

    for (
      let ix = 0;
      ix < GRID_N;
      ix++
    ) {

      const lat =
        GRID_BOUNDS.latMin +
        (
          iy /
          (GRID_N - 1)
        ) *
        (
          GRID_BOUNDS.latMax -
          GRID_BOUNDS.latMin
        );

      const lon =
        GRID_BOUNDS.lonMin +
        (
          ix /
          (GRID_N - 1)
        ) *
        (
          GRID_BOUNDS.lonMax -
          GRID_BOUNDS.lonMin
        );


      points.push({

        lat:
          +lat.toFixed(4),

        lon:
          +lon.toFixed(4),

        ix,
        iy

      });

    }

  }

  return points;

}


async function fetchWithTimeout(
  url,
  options = {},
  ms = 15000
) {

  const controller =
    new AbortController();

  const id =
    setTimeout(
      () =>
        controller.abort(),
      ms
    );

  try {

    return await fetch(
      url,
      {
        ...options,
        signal:
          controller.signal
      }
    );

  } finally {

    clearTimeout(id);

  }

}


/* ---------------- ELEVATION FALLBACK ---------------- */

function elevationFallback(
  lat,
  lon
) {

  const nearestKm =
    Math.min(
      ...DISTRICTS.map(
        d =>
          haversineKm(
            lat,
            lon,
            d.lat,
            d.lon
          )
      )
    );

  return Math.max(
    50,
    2200 -
    nearestKm * 12
  );

}


async function fetchElevationGrid(
  points,
  onProgress
) {

  const chunks = [];

  for (
    let i = 0;
    i < points.length;
    i +=
      ELEV_CHUNK_SIZE
  ) {

    chunks.push(
      points.slice(
        i,
        i +
        ELEV_CHUNK_SIZE
      )
    );

  }


  let degraded =
    false;

  let done =
    0;


  for (
    const chunk of chunks
  ) {

    try {

      const res =
        await fetchWithTimeout(
          'https://api.open-elevation.com/api/v1/lookup',
          {

            method:
              'POST',

            headers:{
              'Content-Type':
                'application/json'
            },

            body:
              JSON.stringify({

                locations:
                  chunk.map(
                    p => ({

                      latitude:
                        p.lat,

                      longitude:
                        p.lon

                    })
                  )

              })

          },
          15000
        );


      if(
        !res.ok
      ) {

        throw new Error(
          'bad response'
        );

      }


      const data =
        await res.json();


      chunk.forEach(
        (
          p,
          i
        ) => {

          p.elevation =
            data.results[i]
              ? data.results[i].elevation
              : elevationFallback(
                  p.lat,
                  p.lon
                );

          p.elevSource =
            'live';

        }
      );


    } catch (
      error
    ) {

      degraded =
        true;


      chunk.forEach(
        p => {

          p.elevation =
            elevationFallback(
              p.lat,
              p.lon
            );

          p.elevSource =
            'estimated';

        }
      );

    }


    done +=
      chunk.length;


    if(
      onProgress
    ){

      onProgress(
        done,
        points.length
      );

    }

  }


  return degraded;

}


function computeSlopes(
  points
) {

  const at =
    (
      ix,
      iy
    ) =>
      points[
        iy * GRID_N +
        ix
      ];


  const clamp =
    v =>
      Math.max(
        0,
        Math.min(
          GRID_N - 1,
          v
        )
      );


  points.forEach(
    p => {

      const pLeft =
        at(
          clamp(
            p.ix - 1
          ),
          p.iy
        );


      const pRight =
        at(
          clamp(
            p.ix + 1
          ),
          p.iy
        );


      const pUp =
        at(
          p.ix,
          clamp(
            p.iy - 1
          )
        );


      const pDown =
        at(
          p.ix,
          clamp(
            p.iy + 1
          )
        );


      const dxKm =
        haversineKm(
          pLeft.lat,
          pLeft.lon,
          pRight.lat,
          pRight.lon
        ) ||
        0.001;


      const dyKm =
        haversineKm(
          pUp.lat,
          pUp.lon,
          pDown.lat,
          pDown.lon
        ) ||
        0.001;


      const dzdx =
        (
          pRight.elevation -
          pLeft.elevation
        ) /
        (
          dxKm *
          1000
        );


      const dzdy =
        (
          pDown.elevation -
          pUp.elevation
        ) /
        (
          dyKm *
          1000
        );


      const slopeRad =
        Math.atan(
          Math.sqrt(
            dzdx * dzdx +
            dzdy * dzdy
          )
        );


      p.slope =
        +Math.max(
          0,
          Math.min(
            60,
            slopeRad *
            180 /
            Math.PI
          )
        ).toFixed(1);

    }
  );

}


/* ---------------- LIVE WEATHER + GROUND SENSOR ---------------- */

async function fetchDistrictWeatherStations() {

  return Promise.all(

    DISTRICTS.map(
      async d => {

        try {

          const res =
            await fetch(
              `/api/weather?lat=${d.lat}&lon=${d.lon}`
            );


          if (
            !res.ok
          ) {

            throw new Error(
              `Weather API ${res.status}`
            );

          }


          const data =
            await res.json();


          return {

            ...d,

            rain:
              Number(
                data.rain_72h_mm ??
                0
              ),

            soil:
              Number(
                data.soil_moisture_pct ??
                45
              ),

            soilSource:
              data.soil_source ||
              'open_meteo',

            soilSensorId:
              data.soil_sensor_id ||
              null,

            live:
              true

          };


        } catch (
          error
        ) {

          console.warn(
            `Weather unavailable for ${d.name}:`,
            error
          );


          return {

            ...d,

            rain:
              0,

            soil:
              45,

            soilSource:
              'fallback',

            soilSensorId:
              null,

            live:
              false

          };

        }

      }
    )

  );

}


function idwInterpolate(
  lat,
  lon,
  stations,
  valueKey
) {

  let weightSum =
    0;

  let valueSum =
    0;


  for (
    const s of stations
  ) {

    const dist =
      Math.max(
        0.01,
        haversineKm(
          lat,
          lon,
          s.lat,
          s.lon
        )
      );


    const weight =
      1 /
      (
        dist *
        dist
      );


    weightSum +=
      weight;


    valueSum +=
      weight *
      s[valueKey];

  }


  return (
    valueSum /
    weightSum
  );

}


/* ============================================================
   BACKEND MODEL SCORING
   ============================================================ */

async function scoreGridWithBackend(
  points
) {

  const res =
    await fetch(
      '/api/predict/grid',
      {

        method:
          'POST',

        headers:{
          'Content-Type':
            'application/json'
        },


        body:
          JSON.stringify({

            points:
              points.map(
                p => ({

                  lat:
                    p.lat,

                  lon:
                    p.lon,

                  rain_72h_mm:
                    p.rain,

                  soil_moisture_pct:
                    p.soil,

                  slope_deg:
                    p.slope

                })
              )

          })

      }
    );


  if (
    !res.ok
  ) {

    const message =
      await res.text();


    throw new Error(
      `predict/grid failed: ${res.status} ${message}`
    );

  }


  const data =
    await res.json();


  if (
    !Array.isArray(
      data.results
    )
  ) {

    throw new Error(
      'Invalid predict/grid response'
    );

  }


  data.results.forEach(
    (
      result,
      i
    ) => {

      points[i].probability =
        Number(
          result.probability ??
          0
        );


      points[i].hazard_index =
        Number(
          result.hazard_index ??
          0
        );


      points[i].seismicInfluence =
        Number(
          result.seismic_influence ??
          0
        );


      points[i].soilSensorId =
        result.soil_sensor_id ??
        null;


      points[i].soilSource =
        result.soil_moisture_source ??
        (
          points[i]
            .soilSensorId

            ? 'ground sensor'

            : 'weather'
        );

    }
  );


  return data;

}


/* ============================================================
   THREE.JS SCENE
   ============================================================ */

const state = {

  scene:
    null,

  camera:
    null,

  renderer:
    null,

  controls:
    null,

  terrainMesh:
    null,

  markerGroup:
    null,

  flowGroup:
    null,

  districtGroup:
    null,

  grid:
    [],

  elevMin:
    0,

  elevMax:
    1,

  exaggeration:
    4,

  hazardBlend:
    0.6,

  markerThreshold:
    0,

  initiationThreshold:
    70,

  agentsPerPoint:
    15,

  wireframe:
    false,

  autoRotate:
    false,

  simStartTime:
    0,

  baseLayer:
    'satellite',

  basemapTexture:
    null,

  raycaster:
    new THREE.Raycaster(),

  pointer:
    new THREE.Vector2()

};


function initScene() {

  const container =
    document.getElementById(
      'viewport'
    );


  const w =
    container.clientWidth;

  const h =
    container.clientHeight;


  state.scene =
    new THREE.Scene();


  state.scene.background =
    new THREE.Color(
      0x0b1220
    );


  state.scene.fog =
    new THREE.Fog(
      0x0b1220,
      WORLD_WIDTH *
      0.9,
      WORLD_WIDTH *
      2.4
    );


  state.camera =
    new THREE.PerspectiveCamera(
      48,
      w / h,
      0.1,
      4000
    );


  setDefaultCameraPosition();


  state.renderer =
    new THREE.WebGLRenderer({
      antialias:
        true
    });


  state.renderer.setPixelRatio(
    Math.min(
      window.devicePixelRatio,
      2
    )
  );


  state.renderer.setSize(
    w,
    h
  );


  state.renderer.outputColorSpace =
    THREE.SRGBColorSpace;


  state.renderer.toneMapping =
    THREE.ACESFilmicToneMapping;


  state.renderer.toneMappingExposure =
    1.15;


  state.renderer.shadowMap.enabled =
    true;


  state.renderer.shadowMap.type =
    THREE.PCFSoftShadowMap;


  container.appendChild(
    state.renderer.domElement
  );


  state.controls =
    new OrbitControls(
      state.camera,
      state.renderer.domElement
    );


  state.controls.enableDamping =
    true;


  state.controls.dampingFactor =
    0.08;


  state.controls.minDistance =
    40;


  state.controls.maxDistance =
    WORLD_WIDTH *
    2.2;


  state.controls.maxPolarAngle =
    Math.PI *
    0.49;


  state.controls.target.set(
    0,
    0,
    0
  );


  state.controls.autoRotateSpeed =
    1.1;


  const hemi =
    new THREE.HemisphereLight(
      0xbfd9ff,
      0x11161f,
      0.55
    );


  state.scene.add(
    hemi
  );


  const sun =
    new THREE.DirectionalLight(
      0xfff3e2,
      2.0
    );


  sun.position.set(
    WORLD_WIDTH *
    0.55,

    WORLD_WIDTH *
    0.38,

    -WORLD_DEPTH *
    0.35
  );


  sun.castShadow =
    true;


  const shadowExtent =
    Math.max(
      WORLD_WIDTH,
      WORLD_DEPTH
    ) *
    0.75;


  sun.shadow.camera.left =
    -shadowExtent;

  sun.shadow.camera.right =
    shadowExtent;

  sun.shadow.camera.top =
    shadowExtent;

  sun.shadow.camera.bottom =
    -shadowExtent;


  sun.shadow.camera.near =
    1;


  sun.shadow.camera.far =
    WORLD_WIDTH *
    3;


  sun.shadow.mapSize.set(
    2048,
    2048
  );


  sun.shadow.bias =
    -0.0015;


  sun.shadow.normalBias =
    0.4;


  state.scene.add(
    sun
  );

  state.scene.add(
    sun.target
  );


  const fill =
    new THREE.AmbientLight(
      0x2a3548,
      0.55
    );


  state.scene.add(
    fill
  );


  const bounce =
    new THREE.DirectionalLight(
      0x5f7fa8,
      0.35
    );


  bounce.position.set(
    -WORLD_WIDTH *
    0.4,

    WORLD_WIDTH *
    0.2,

    WORLD_DEPTH *
    0.5
  );


  state.scene.add(
    bounce
  );


  state.markerGroup =
    new THREE.Group();

  state.flowGroup =
    new THREE.Group();

  state.districtGroup =
    new THREE.Group();


  state.scene.add(
    state.markerGroup,
    state.flowGroup,
    state.districtGroup
  );


  window.addEventListener(
    'resize',
    onResize
  );


  state.renderer
    .domElement
    .addEventListener(
      'pointerdown',
      onPointerDown
    );


  animate();

}


function setDefaultCameraPosition() {

  state.camera.position.set(
    WORLD_WIDTH *
    0.55,

    WORLD_WIDTH *
    0.55,

    WORLD_DEPTH *
    0.65
  );


  state.camera.lookAt(
    0,
    0,
    0
  );

}


function onResize() {

  const container =
    document.getElementById(
      'viewport'
    );


  const w =
    container.clientWidth;

  const h =
    container.clientHeight;


  state.camera.aspect =
    w /
    h;


  state.camera.updateProjectionMatrix();


  state.renderer.setSize(
    w,
    h
  );

}


function animate() {

  requestAnimationFrame(
    animate
  );


  state.controls.update();


  state.renderer.render(
    state.scene,
    state.camera
  );

}


/* ---------------- TERRAIN MESH ---------------- */

let _renderElevCache =
  null;


function buildRenderElevationField() {

  const N =
    RENDER_N;


  const elev =
    new Float32Array(
      N * N
    );


  const noiseFreq =
    N / 14;


  for (
    let iy = 0;
    iy < N;
    iy++
  ) {

    for (
      let ix = 0;
      ix < N;
      ix++
    ) {

      const fx =
        (
          ix /
          (N - 1)
        ) *
        (
          GRID_N - 1
        );


      const fy =
        (
          iy /
          (N - 1)
        ) *
        (
          GRID_N - 1
        );


      const smooth =
        bilinearElevation(
          fx,
          fy
        );


      elev[
        iy * N + ix
      ] =
        smooth;

    }

  }


  const elevMin =
    Math.min(
      ...elev
    );

  const elevMax =
    Math.max(
      ...elev
    );


  const relief =
    Math.max(
      1,
      elevMax -
      elevMin
    );


  for (
    let iy = 0;
    iy < N;
    iy++
  ) {

    for (
      let ix = 0;
      ix < N;
      ix++
    ) {

      const n =
        fbmNoise2(
          ix /
          noiseFreq,

          iy /
          noiseFreq
        );


      elev[
        iy * N + ix
      ] +=
        n *
        relief *
        0.05;

    }

  }


  return {

    elev,

    elevMin,

    elevMax:
      Math.max(
        ...elev
      )

  };

}


/* ---------------- FLOW ACCUMULATION ---------------- */

function computeFlowAccumulation(
  elev,
  N
) {

  const order = [];


  for (
    let i = 0;
    i < N * N;
    i++
  ) {

    order.push(i);

  }


  order.sort(
    (
      a,
      b
    ) =>
      elev[b] -
      elev[a]
  );


  const accum =
    new Float32Array(
      N * N
    ).fill(
      1
    );


  const dirs = [
    -1,
    0,
    1
  ];


  for (
    const idx of order
  ) {

    const ix =
      idx %
      N;


    const iy =
      (
        idx /
        N
      ) |
      0;


    let bestJ =
      -1;


    let bestDrop =
      0;


    for (
      const dy
      of dirs
    ) {

      for (
        const dx
        of dirs
      ) {

        if (
          dx === 0 &&
          dy === 0
        ) {

          continue;

        }


        const nx =
          ix +
          dx;

        const ny =
          iy +
          dy;


        if (
          nx < 0 ||
          ny < 0 ||
          nx >= N ||
          ny >= N
        ) {

          continue;

        }


        const nIdx =
          ny * N +
          nx;


        const drop =
          elev[idx] -
          elev[nIdx];


        if (
          drop >
          bestDrop
        ) {

          bestDrop =
            drop;

          bestJ =
            nIdx;

        }

      }

    }


    if (
      bestJ >= 0
    ) {

      accum[bestJ] +=
        accum[idx];

    }

  }


  let maxA =
    1;


  for (
    let i = 0;
    i < accum.length;
    i++
  ) {

    if (
      accum[i] >
      maxA
    ) {

      maxA =
        accum[i];

    }

  }


  const logMax =
    Math.log1p(
      maxA
    );


  const river =
    new Float32Array(
      N * N
    );


  for (
    let i = 0;
    i < accum.length;
    i++
  ) {

    river[i] =
      Math.log1p(
        accum[i]
      ) /
      logMax;

  }


  return river;

}


/* ---------------- BASEMAP IMAGERY ---------------- */

let _basemapLoadToken =
  0;


async function buildBasemapTexture(
  layer,
  onProgress
) {

  if (
    layer === 'shading'
  ) {

    return null;

  }


  const myToken =
    ++_basemapLoadToken;


  const canvas =
    document.createElement(
      'canvas'
    );


  canvas.width =
    TILE_SPAN_X *
    TILE_PX;


  canvas.height =
    TILE_SPAN_Y *
    TILE_PX;


  const ctx =
    canvas.getContext(
      '2d'
    );


  ctx.fillStyle =
    '#1b2637';


  ctx.fillRect(
    0,
    0,
    canvas.width,
    canvas.height
  );


  const jobs = [];


  for (
    let ty =
      TILE_Y_MIN;
    ty <= TILE_Y_MAX;
    ty++
  ) {

    for (
      let tx =
        TILE_X_MIN;
      tx <= TILE_X_MAX;
      tx++
    ) {

      jobs.push({
        tx,
        ty
      });

    }

  }


  let done =
    0;


  let failures =
    0;


  await Promise.all(

    jobs.map(
      (
        {
          tx,
          ty
        }
      ) =>

        new Promise(
          resolve => {

            const img =
              new Image();


            img.crossOrigin =
              'anonymous';


            img.onload =
              () => {

                if (
                  myToken ===
                  _basemapLoadToken
                ) {

                  ctx.drawImage(
                    img,

                    (
                      tx -
                      TILE_X_MIN
                    ) *
                    TILE_PX,

                    (
                      ty -
                      TILE_Y_MIN
                    ) *
                    TILE_PX,

                    TILE_PX,

                    TILE_PX
                  );

                }


                done++;


                if (
                  onProgress
                ) {

                  onProgress(
                    done,
                    jobs.length
                  );

                }


                resolve();

              };


            img.onerror =
              () => {

                failures++;

                done++;


                if (
                  onProgress
                ) {

                  onProgress(
                    done,
                    jobs.length
                  );

                }


                resolve();

              };


            img.src =
              `/api/tile/${layer}/${TILE_ZOOM}/${tx}/${ty}`;

          }
        )

    )

  );


  if (
    myToken !==
    _basemapLoadToken
  ) {

    return null;

  }


  if (
    failures ===
    jobs.length
  ) {

    return null;

  }


  const texture =
    new THREE.CanvasTexture(
      canvas
    );


  texture.colorSpace =
    THREE.SRGBColorSpace;


  texture.wrapS =
    THREE.ClampToEdgeWrapping;


  texture.wrapT =
    THREE.ClampToEdgeWrapping;


  texture.minFilter =
    THREE.LinearMipmapLinearFilter;


  texture.magFilter =
    THREE.LinearFilter;


  texture.generateMipmaps =
    true;


  texture.needsUpdate =
    true;


  if (
    failures > 0
  ) {

    toast(
      `${failures} of ${jobs.length} basemap tiles failed to load.`,
      'warn',
      5000
    );

  }


  return texture;

}


async function setBaseLayer(
  layer
) {

  state.baseLayer =
    layer;


  const oldTexture =
    state.basemapTexture;


  if (
    layer ===
    'shading'
  ) {

    state.basemapTexture =
      null;


    oldTexture?.dispose();


    if (
      state.grid.length
    ) {

      buildTerrainMesh();

    }


    return;

  }


  toast(
    layer === 'satellite'
      ? 'Loading real satellite imagery…'
      : 'Loading real topographic map tiles…',
    'info',
    3000
  );


  const tex =
    await buildBasemapTexture(
      layer
    );


  if (
    !tex
  ) {

    toast(
      'Could not load basemap tiles — showing procedural terrain shading instead.',
      'warn',
      5000
    );

  }


  state.basemapTexture =
    tex;


  oldTexture?.dispose();


  if (
    state.grid.length
  ) {

    buildTerrainMesh();

  }

}


function buildTerrainMesh() {

  if (
    state.terrainMesh
  ) {

    state.scene.remove(
      state.terrainMesh
    );


    state.terrainMesh.geometry.dispose();


    state.terrainMesh.material.dispose();

  }


  const N =
    RENDER_N;


  if (
    !_renderElevCache
  ) {

    _renderElevCache =
      buildRenderElevationField();

  }


  const {
    elev,
    elevMin,
    elevMax
  } =
    _renderElevCache;


  const elevRange =
    Math.max(
      1,
      elevMax -
      elevMin
    );


  if (
    !_renderElevCache.river
  ) {

    _renderElevCache.river =
      computeFlowAccumulation(
        elev,
        N
      );

  }


  const river =
    _renderElevCache.river;


  state.elevMin =
    Math.min(
      ...state.grid.map(
        p =>
          p.elevation
      )
    );


  state.elevMax =
    Math.max(
      ...state.grid.map(
        p =>
          p.elevation
      )
    );


  const geometry =
    new THREE.BufferGeometry();


  const positions =
    new Float32Array(
      N * N * 3
    );


  const colors =
    new Float32Array(
      N * N * 3
    );


  const uvs =
    new Float32Array(
      N * N * 2
    );


  const hasImagery =
    !!state.basemapTexture;


  const riverColor =
    new THREE.Color(
      0x3fa9dd
    );


  const hazardColor =
    new THREE.Color(
      0xff5a3c
    );


  const tmp =
    new THREE.Color();


  for (
    let iy = 0;
    iy < N;
    iy++
  ) {

    for (
      let ix = 0;
      ix < N;
      ix++
    ) {

      const idx =
        iy * N +
        ix;


      const fx =
        (
          ix /
          (N - 1)
        ) *
        (
          GRID_N - 1
        );


      const fy =
        (
          iy /
          (N - 1)
        ) *
        (
          GRID_N - 1
        );


      const lat =
        GRID_BOUNDS.latMin +
        (
          fy /
          (
            GRID_N -
            1
          )
        ) *
        (
          GRID_BOUNDS.latMax -
          GRID_BOUNDS.latMin
        );


      const lon =
        GRID_BOUNDS.lonMin +
        (
          fx /
          (
            GRID_N -
            1
          )
        ) *
        (
          GRID_BOUNDS.lonMax -
          GRID_BOUNDS.lonMin
        );


      const {
        x,
        z
      } =
        latLonToXZ(
          lat,
          lon
        );


      const y =
        elev[idx] *
        ELEV_BASE_SCALE *
        state.exaggeration;


      const pIdx =
        idx * 3;


      positions[pIdx] =
        x;


      positions[
        pIdx + 1
      ] =
        y;


      positions[
        pIdx + 2
      ] =
        z;


      const uvIdx =
        idx * 2;


      const {
        u,
        v
      } =
        basemapUV(
          lat,
          lon
        );


      uvs[
        uvIdx
      ] =
        u;


      uvs[
        uvIdx + 1
      ] =
        1 -
        v;


      const xm =
        elev[
          iy * N +
          Math.max(
            0,
            ix - 1
          )
        ];


      const xp =
        elev[
          iy * N +
          Math.min(
            N - 1,
            ix + 1
          )
        ];


      const ym =
        elev[
          Math.max(
            0,
            iy - 1
          ) *
          N +
          ix
        ];


      const yp =
        elev[
          Math.min(
            N - 1,
            iy + 1
          ) *
          N +
          ix
        ];


      const dzdx =
        (
          xp -
          xm
        ) /
        (
          2 *
          RENDER_CELL_METERS_X
        );


      const dzdy =
        (
          yp -
          ym
        ) /
        (
          2 *
          RENDER_CELL_METERS_Y
        );


      const slopeDeg =
        Math.atan(
          Math.hypot(
            dzdx,
            dzdy
          )
        ) *
        180 /
        Math.PI;


      const normElev =
        (
          elev[idx] -
          elevMin
        ) /
        elevRange;


      if (
        hasImagery
      ) {

        tmp.setRGB(
          1,
          1,
          1
        );

      } else {

        tmp.copy(
          terrainAlbedo(
            normElev,
            slopeDeg
          )
        );

      }


      const riverT =
        Math.max(
          0,
          (
            river[idx] -
            0.86
          ) /
          0.14
        );


      if (
        riverT > 0
      ) {

        tmp.lerp(
          riverColor,
          Math.min(
            1,
            riverT
          ) *
          (
            hasImagery
              ? 0.55
              : 0.85
          )
        );

      }


      const prob =
        bicubicProbability(
          fx,
          fy
        );


      const hazardT =
        Math.pow(
          prob,
          1.6
        ) *
        state.hazardBlend;


      if (
        hazardT > 0.01
      ) {

        tmp.lerp(
          hazardColor,
          Math.min(
            hasImagery
              ? 0.75
              : 0.9,

            hazardT
          )
        );

      }


      colors[pIdx] =
        tmp.r;

      colors[
        pIdx + 1
      ] =
        tmp.g;

      colors[
        pIdx + 2
      ] =
        tmp.b;

    }

  }


  const indices =
    [];


  for (
    let iy = 0;
    iy < N - 1;
    iy++
  ) {

    for (
      let ix = 0;
      ix < N - 1;
      ix++
    ) {

      const a =
        iy * N +
        ix;

      const b =
        iy * N +
        ix + 1;

      const c =
        (
          iy + 1
        ) *
        N +
        ix;

      const d =
        (
          iy + 1
        ) *
        N +
        ix +
        1;


      indices.push(
        a,
        c,
        b,

        b,
        c,
        d
      );

    }

  }


  geometry.setIndex(
    indices
  );


  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(
      positions,
      3
    )
  );


  geometry.setAttribute(
    'color',
    new THREE.BufferAttribute(
      colors,
      3
    )
  );


  geometry.setAttribute(
    'uv',
    new THREE.BufferAttribute(
      uvs,
      2
    )
  );


  geometry.computeVertexNormals();


  const material =
    new THREE.MeshStandardMaterial({

      vertexColors:
        true,

      side:
        THREE.DoubleSide,

      wireframe:
        state.wireframe,

      roughness:
        hasImagery
          ? 0.88
          : 0.96,

      metalness:
        0.0,

      flatShading:
        false,

      map:
        state.basemapTexture ||
        null

    });


  state.terrainMesh =
    new THREE.Mesh(
      geometry,
      material
    );


  state.terrainMesh.castShadow =
    true;


  state.terrainMesh.receiveShadow =
    true;


  state.scene.add(
    state.terrainMesh
  );


  buildDistrictLabels();

}


function heightAt(
  ix,
  iy
) {

  return (
    bilinearElevation(
      ix,
      iy
    ) *
    ELEV_BASE_SCALE *
    state.exaggeration
  );

}


function buildDistrictLabels() {

  state.districtGroup.clear();


  DISTRICTS.forEach(
    d => {

      const {
        x,
        z
      } =
        latLonToXZ(
          d.lat,
          d.lon
        );


      const y =
        terrainHeightAtLatLon(
          d.lat,
          d.lon
        ) +
        2;


      const dot =
        new THREE.Mesh(
          new THREE.SphereGeometry(
            1.6,
            12,
            12
          ),

          new THREE.MeshBasicMaterial({
            color:
              0x64748b
          })
        );


      dot.position.set(
        x,
        y,
        z
      );


      state.districtGroup.add(
        dot
      );

    }
  );

}


function terrainHeightAtLatLon(
  lat,
  lon
) {

  const ix =
    Math.round(
      (
        lon -
        GRID_BOUNDS.lonMin
      ) /
      (
        GRID_BOUNDS.lonMax -
        GRID_BOUNDS.lonMin
      ) *
      (
        GRID_N -
        1
      )
    );


  const iy =
    Math.round(
      (
        lat -
        GRID_BOUNDS.latMin
      ) /
      (
        GRID_BOUNDS.latMax -
        GRID_BOUNDS.latMin
      ) *
      (
        GRID_N -
        1
      )
    );


  const cix =
    Math.max(
      0,
      Math.min(
        GRID_N - 1,
        ix
      )
    );


  const ciy =
    Math.max(
      0,
      Math.min(
        GRID_N - 1,
        iy
      )
    );


  return heightAt(
    cix,
    ciy
  );

}


/* ---------------- HAZARD MARKERS ---------------- */

let markerTexture =
  null;


function getMarkerTexture() {

  if (
    markerTexture
  ) {

    return markerTexture;

  }


  const size =
    64;


  const canvas =
    document.createElement(
      'canvas'
    );


  canvas.width =
    canvas.height =
    size;


  const ctx =
    canvas.getContext(
      '2d'
    );


  const grad =
    ctx.createRadialGradient(
      size / 2,
      size / 2,
      0,
      size / 2,
      size / 2,
      size / 2
    );


  grad.addColorStop(
    0,
    'rgba(255,255,255,1)'
  );


  grad.addColorStop(
    0.55,
    'rgba(255,255,255,0.9)'
  );


  grad.addColorStop(
    1,
    'rgba(255,255,255,0)'
  );


  ctx.fillStyle =
    grad;


  ctx.beginPath();


  ctx.arc(
    size / 2,
    size / 2,
    size / 2,
    0,
    Math.PI * 2
  );


  ctx.fill();


  markerTexture =
    new THREE.CanvasTexture(
      canvas
    );


  return markerTexture;

}


function buildHazardMarkers() {

  state.markerGroup.clear();


  const texture =
    getMarkerTexture();


  let count =
    0;


  state.grid.forEach(
    p => {

      const pct =
        (
          p.probability ||
          0
        ) *
        100;


      if (
        pct <
        state.markerThreshold
      ) {

        return;

      }


      count++;


      const {
        x,
        z
      } =
        latLonToXZ(
          p.lat,
          p.lon
        );


      const y =
        p.elevation *
        ELEV_BASE_SCALE *
        state.exaggeration;


      const color =
        new THREE.Color(
          riskColorCss(
            pct
          )
        );


      const mat =
        new THREE.SpriteMaterial({

          map:
            texture,

          color,

          depthWrite:
            false,

          transparent:
            true

        });


      const sprite =
        new THREE.Sprite(
          mat
        );


      const scale =
        3.5 +
        (
          pct /
          100
        ) *
        5.5;


      sprite.scale.set(
        scale,
        scale,
        1
      );


      sprite.position.set(
        x,
        y +
        scale *
        0.55,
        z
      );


      sprite.userData =
        {
          point:
            p
        };


      state.markerGroup.add(
        sprite
      );

    }
  );


  document.getElementById(
    'statCells'
  ).textContent =
    state.grid.length;


  return count;

}


/* ---------------- FLOW SIMULATION ---------------- */

function worldPosAtFractionalCell(
  fx,
  fy
) {

  const cx =
    Math.max(
      0,
      Math.min(
        GRID_N - 1,
        fx
      )
    );


  const cy =
    Math.max(
      0,
      Math.min(
        GRID_N - 1,
        fy
      )
    );


  const lat =
    GRID_BOUNDS.latMin +
    (
      cy /
      (
        GRID_N - 1
      )
    ) *
    (
      GRID_BOUNDS.latMax -
      GRID_BOUNDS.latMin
    );


  const lon =
    GRID_BOUNDS.lonMin +
    (
      cx /
      (
        GRID_N - 1
      )
    ) *
    (
      GRID_BOUNDS.lonMax -
      GRID_BOUNDS.lonMin
    );


  const {
    x,
    z
  } =
    latLonToXZ(
      lat,
      lon
    );


  const y =
    bilinearElevation(
      fx,
      fy
    ) *
    ELEV_BASE_SCALE *
    state.exaggeration;


  return new THREE.Vector3(
    x,
    y,
    z
  );

}


function simulateAgentPath(
  startIx,
  startIy,
  seed
) {

  const rand =
    mulberry32(
      seed
    );


  let fx =
    startIx +
    (
      rand() -
      0.5
    ) *
    0.5;


  let fy =
    startIy +
    (
      rand() -
      0.5
    ) *
    0.5;


  const path = [
    worldPosAtFractionalCell(
      fx,
      fy
    )
  ];


  const step =
    0.55;


  const eps =
    0.35;


  let flatStreak =
    0;


  for (
    let i = 0;
    i < 45;
    i++
  ) {

    if (
      fx < 0.3 ||
      fx >
        GRID_N -
        1.3 ||
      fy < 0.3 ||
      fy >
        GRID_N -
        1.3
    ) {

      break;

    }


    const eC =
      bilinearElevation(
        fx,
        fy
      );


    const eXp =
      bilinearElevation(
        fx + eps,
        fy
      );


    const eXm =
      bilinearElevation(
        fx - eps,
        fy
      );


    const eYp =
      bilinearElevation(
        fx,
        fy + eps
      );


    const eYm =
      bilinearElevation(
        fx,
        fy - eps
      );


    let gx =
      (
        eXp -
        eXm
      ) /
      (
        2 *
        eps
      );


    let gy =
      (
        eYp -
        eYm
      ) /
      (
        2 *
        eps
      );


    const gradMag =
      Math.hypot(
        gx,
        gy
      );


    if (
      gradMag <
      0.15
    ) {

      flatStreak++;


      if (
        flatStreak >
        3
      ) {

        break;

      }

    } else {

      flatStreak =
        0;

    }


    let dx =
      -gx;


    let dy =
      -gy;


    const mag =
      Math.hypot(
        dx,
        dy
      ) ||
      1;


    dx /=
      mag;


    dy /=
      mag;


    const jitter =
      (
        rand() -
        0.5
      ) *
      0.6;


    const px =
      -dy;


    const py =
      dx;


    dx +=
      px *
      jitter;


    dy +=
      py *
      jitter;


    const finalMag =
      Math.hypot(
        dx,
        dy
      ) ||
      1;


    fx +=
      (
        dx /
        finalMag
      ) *
      step;


    fy +=
      (
        dy /
        finalMag
      ) *
      step;


    path.push(
      worldPosAtFractionalCell(
        fx,
        fy
      )
    );


    if (
      eC <=
      state.elevMin +
      (
        state.elevMax -
        state.elevMin
      ) *
      0.04
    ) {

      break;

    }

  }


  return path;

}


function mulberry32(
  a
) {

  return function () {

    a |=
      0;


    a =
      (
        a +
        0x6D2B79F5
      ) |
      0;


    let t =
      Math.imul(
        a ^
        (
          a >>>
          15
        ),

        1 |
        a
      );


    t =
      (
        t +
        Math.imul(
          t ^
          (
            t >>>
            7
          ),

          61 |
          t
        )
      ) ^
      t;


    return (
      (
        t ^
        (
          t >>>
          14
        )
      ) >>>
      0
    ) /
    4294967296;

  };

}


/* ---------------- FLOW GRADIENT ---------------- */

let _flowGradientTex =
  null;


function getFlowGradientTexture() {

  if (
    _flowGradientTex
  ) {

    return _flowGradientTex;

  }


  const w =
    256;


  const h =
    8;


  const canvas =
    document.createElement(
      'canvas'
    );


  canvas.width =
    w;


  canvas.height =
    h;


  const ctx =
    canvas.getContext(
      '2d'
    );


  const grad =
    ctx.createLinearGradient(
      0,
      0,
      w,
      0
    );


  grad.addColorStop(
    0.0,
    '#38bdf8'
  );


  grad.addColorStop(
    0.45,
    '#22d3ee'
  );


  grad.addColorStop(
    0.75,
    '#f59e0b'
  );


  grad.addColorStop(
    1.0,
    '#ef4444'
  );


  ctx.fillStyle =
    grad;


  ctx.fillRect(
    0,
    0,
    w,
    h
  );


  _flowGradientTex =
    new THREE.CanvasTexture(
      canvas
    );


  return _flowGradientTex;

}


function disposeFlowGroup() {

  state.flowGroup.children.forEach(
    obj => {

      obj.geometry?.dispose();

      obj.material?.dispose();

    }
  );


  state.flowGroup.clear();

}


let simAnimHandle =
  null;


function runFlowSimulation() {

  disposeFlowGroup();


  if (
    simAnimHandle
  ) {

    cancelAnimationFrame(
      simAnimHandle
    );

  }


  const initiationPoints =
    state.grid.filter(
      p =>
        (
          p.probability ||
          0
        ) *
        100 >=
        state.initiationThreshold
    );


  document.getElementById(
    'simStatus'
  ).textContent =
    'RUNNING';


  document.getElementById(
    'simStatus'
  ).className =
    'mono font-bold text-amber-400 mb-2 tracking-wide';


  document.getElementById(
    'statInit'
  ).textContent =
    initiationPoints.length;


  document.getElementById(
    'statSlides'
  ).textContent =
    initiationPoints.length;


  document.getElementById(
    'statAgents'
  ).textContent =
    initiationPoints.length *
    state.agentsPerPoint;


  document.getElementById(
    'statTime'
  ).textContent =
    '--';


  if (
    !document.getElementById(
      'toggleFlow'
    ).checked
  ) {

    document.getElementById(
      'simStatus'
    ).textContent =
      'HIDDEN';


    document.getElementById(
      'simStatus'
    ).className =
      'mono font-bold text-slate-500 mb-2 tracking-wide';


    return;

  }


  state.simStartTime =
    performance.now();


  const tubes =
    [];


  const gradientTex =
    getFlowGradientTexture();


  let seed =
    1;


  initiationPoints.forEach(
    p => {

      for (
        let a = 0;
        a <
        state.agentsPerPoint;
        a++
      ) {

        const path =
          simulateAgentPath(
            p.ix,
            p.iy,
            seed++
          );


        if (
          path.length <
          3
        ) {

          continue;

        }


        const raised =
          path.map(
            v =>
              new THREE.Vector3(
                v.x,
                v.y +
                0.7,
                v.z
              )
          );


        const curve =
          new THREE.CatmullRomCurve3(
            raised
          );


        const tubularSegments =
          Math.max(
            8,
            raised.length *
            3
          );


        const radius =
          0.55 +
          (
            state.exaggeration /
            10
          ) *
          0.25;


        const coreGeo =
          new THREE.TubeGeometry(
            curve,
            tubularSegments,
            radius,
            6,
            false
          );


        const coreMat =
          new THREE.MeshBasicMaterial({

            map:
              gradientTex,

            transparent:
              true,

            opacity:
              0.95,

            depthWrite:
              false

          });


        const core =
          new THREE.Mesh(
            coreGeo,
            coreMat
          );


        state.flowGroup.add(
          core
        );


        const glowGeo =
          new THREE.TubeGeometry(
            curve,
            tubularSegments,
            radius *
              2.6,
            6,
            false
          );


        const glowMat =
          new THREE.MeshBasicMaterial({

            map:
              gradientTex,

            transparent:
              true,

            opacity:
              0.28,

            depthWrite:
              false,

            blending:
              THREE.AdditiveBlending

          });


        const glow =
          new THREE.Mesh(
            glowGeo,
            glowMat
          );


        state.flowGroup.add(
          glow
        );


        const indexCount =
          coreGeo.index.count;


        core.geometry.setDrawRange(
          0,
          0
        );


        glow.geometry.setDrawRange(
          0,
          0
        );


        tubes.push({

          core,

          glow,

          indexCount

        });

      }

    }
  );


  const animStart =
    performance.now();


  const durationMs =
    1800;


  function step() {

    const elapsed =
      performance.now() -
      animStart;


    const t =
      Math.min(
        1,
        elapsed /
        durationMs
      );


    tubes.forEach(
      ({
        core,
        glow,
        indexCount
      }) => {

        const count =
          Math.ceil(
            indexCount *
            t /
            6
          ) *
          6;


        core.geometry.setDrawRange(
          0,
          count
        );


        glow.geometry.setDrawRange(
          0,
          count
        );

      }
    );


    if (
      t < 1
    ) {

      simAnimHandle =
        requestAnimationFrame(
          step
        );

    } else {

      const totalMs =
        Math.round(
          performance.now() -
          state.simStartTime
        );


      document.getElementById(
        'simStatus'
      ).textContent =
        'COMPLETED';


      document.getElementById(
        'simStatus'
      ).className =
        'mono font-bold text-emerald-400 mb-2 tracking-wide';


      document.getElementById(
        'statTime'
      ).textContent =
        (
          totalMs /
          1000
        ).toFixed(
          2
        ) +
        's';

    }

  }


  simAnimHandle =
    requestAnimationFrame(
      step
    );

}


/* ---------------- PICKING ---------------- */

function onPointerDown(
  event
) {

  const rect =
    state.renderer
      .domElement
      .getBoundingClientRect();


  state.pointer.x =
    (
      (
        event.clientX -
        rect.left
      ) /
      rect.width
    ) *
    2 -
    1;


  state.pointer.y =
    -(
      (
        event.clientY -
        rect.top
      ) /
      rect.height
    ) *
    2 +
    1;


  state.raycaster.setFromCamera(
    state.pointer,
    state.camera
  );


  const hits =
    state.raycaster.intersectObjects(
      state.markerGroup.children,
      false
    );


  if (
    hits.length
  ) {

    showDetail(
      hits[0]
        .object
        .userData
        .point
    );

  }

}


function showDetail(
  p
) {

  const pct =
    Math.round(
      (
        p.probability ||
        0
      ) *
      100
    );


  const color =
    riskColorCss(
      pct
    );


  const seismicPct =
    Math.round(
      Number(
        p.seismicInfluence ??
        0
      ) *
      100
    );


  const soilSource =
    p.soilSensorId

      ? `Ground sensor · ${p.soilSensorId}`

      : (
          p.soilSource ||
          'Weather-derived'
        );


  document.getElementById(
    'detailBody'
  ).innerHTML = `

    <div class="text-slate-400">
      ${p.lat.toFixed(3)},
      ${p.lon.toFixed(3)}
    </div>

    <div class="text-slate-400 mb-1">
      Nearest:

      <span class="text-slate-200">

        ${districtName(
          nearestDistrictId(
            p.lat,
            p.lon
          )
        )}

      </span>

    </div>

    <div class="grid grid-cols-2 gap-x-3 gap-y-0.5">

      <span class="text-slate-400">
        Elevation
      </span>

      <span>
        ${Math.round(p.elevation)} m

        ${
          p.elevSource === 'live'
            ? '<span class="text-emerald-500">(live)</span>'
            : '<span class="text-amber-500">(est.)</span>'
        }

      </span>


      <span class="text-slate-400">
        Slope
      </span>

      <span>
        ${Number(
          p.slope ||
          0
        ).toFixed(
          1
        )}°
      </span>


      <span class="text-slate-400">
        Rain (72h)
      </span>

      <span>
        ${
          p.rain != null
            ? Number(
                p.rain
              ).toFixed(
                0
              )
            : '--'
        } mm
      </span>


      <span class="text-slate-400">
        Soil Moisture
      </span>

      <span>
        ${
          p.soil != null
            ? Number(
                p.soil
              ).toFixed(
                0
              )
            : '--'
        }%
      </span>


      <span class="text-slate-400">
        Soil Source
      </span>

      <span>
        ${soilSource}
      </span>


      <span class="text-slate-400">
        Seismic Influence
      </span>

      <span>
        ${seismicPct}%
      </span>

    </div>


    <div class="mt-2 pt-2 border-t border-slate-700 flex items-center justify-between">

      <span class="text-slate-400">
        P<sub>failure</sub>
      </span>

      <span
        class="font-bold"
        style="color:${color}"
      >
        ${pct}%
      </span>

    </div>

  `;


  document.getElementById(
    'detailPanel'
  ).classList.remove(
    'hidden'
  );

}


document
  .getElementById(
    'detailClose'
  )
  .addEventListener(
    'click',
    () =>
      document
        .getElementById(
          'detailPanel'
        )
        .classList.add(
          'hidden'
        )
  );


/* ============================================================
   LOADING / ORCHESTRATION
   ============================================================ */

function setLoading(
  text,
  sub,
  pct
) {

  document.getElementById(
    'loadingText'
  ).textContent =
    text;


  document.getElementById(
    'loadingSub'
  ).textContent =
    sub ||
    '';


  document.getElementById(
    'loadingBar'
  ).style.width =
    Math.max(
      0,
      Math.min(
        100,
        pct
      )
    ) +
    '%';

}


async function loadAndBuildAll() {

  const overlay =
    document.getElementById(
      'loadingOverlay'
    );


  overlay.classList.remove(
    'hidden'
  );


  try {

    setLoading(
      'Fetching elevation grid (Open-Elevation)…',

      `0 / ${
        GRID_N *
        GRID_N
      } points`,

      5
    );


    _renderElevCache =
      null;


    const points =
      buildGridPoints();


    const degraded =
      await fetchElevationGrid(
        points,
        (
          done,
          total
        ) => {

          setLoading(

            'Fetching elevation grid (Open-Elevation)…',

            `${done} / ${total} points`,

            5 +
            (
              done /
              total
            ) *
            55

          );

        }
      );


    if (
      degraded
    ) {

      toast(
        'Open-Elevation was partially unreachable — some cells use an estimated elevation instead of real data.',
        'warn',
        6000
      );

    }


    computeSlopes(
      points
    );


    state.grid =
      points;


    setLoading(
      'Fetching live rainfall, soil sensors & weather…',
      '',
      65
    );


    const stations =
      await fetchDistrictWeatherStations();


    state.grid.forEach(
      p => {

        p.rain =
          +idwInterpolate(
            p.lat,
            p.lon,
            stations,
            'rain'
          ).toFixed(
            1
          );


        p.soil =
          +idwInterpolate(
            p.lat,
            p.lon,
            stations,
            'soil'
          ).toFixed(
            1
          );

      }
    );


    setLoading(
      'Scoring hazard model across the grid…',
      '',
      80
    );


    const prediction =
      await scoreGridWithBackend(
        state.grid
      );


    if (
      state.baseLayer !==
      'shading'
    ) {

      setLoading(
        'Loading real satellite/topo imagery tiles…',

        `0 / ${
          TILE_SPAN_X *
          TILE_SPAN_Y
        } tiles`,

        85
      );


      const tex =
        await buildBasemapTexture(
          state.baseLayer,
          (
            done,
            total
          ) => {

            setLoading(

              'Loading real satellite/topo imagery tiles…',

              `${done} / ${total} tiles`,

              85 +
              (
                done /
                total
              ) *
              10

            );

          }
        );


      state.basemapTexture =
        tex;


      if (
        !tex
      ) {

        toast(
          'Could not load basemap tiles — showing procedural terrain shading instead.',
          'warn',
          5000
        );

      }

    }


    setLoading(
      'Building 3D terrain…',
      '',
      96
    );


    buildTerrainMesh();


    const markerCount =
      buildHazardMarkers();


    toast(
      `${markerCount} hazardous points marked out of ${state.grid.length} grid cells.`,
      'info',
      5000
    );


    runFlowSimulation();


    setLoading(
      'Done',
      '',
      100
    );


    setTimeout(
      () =>
        overlay.classList.add(
          'hidden'
        ),
      300
    );


  } catch (
    e
  ) {

    console.error(
      e
    );


    setLoading(
      'Something went wrong building the 3D view.',
      String(
        e.message ||
        e
      ),
      100
    );


    toast(
      'Could not build the 3D terrain — check that the backend is reachable.',
      'danger',
      8000
    );

  }

}


async function refreshLiveDataOnly() {

  if (
    !state.grid.length
  ) {

    return;

  }


  try {

    toast(
      'Refreshing live rainfall, soil, seismic risk and ground data…',
      'info',
      3000
    );


    const stations =
      await fetchDistrictWeatherStations();


    state.grid.forEach(
      p => {

        p.rain =
          +idwInterpolate(
            p.lat,
            p.lon,
            stations,
            'rain'
          ).toFixed(
            1
          );


        p.soil =
          +idwInterpolate(
            p.lat,
            p.lon,
            stations,
            'soil'
          ).toFixed(
            1
          );

      }
    );


    const prediction =
      await scoreGridWithBackend(
        state.grid
      );


    buildTerrainMesh();


    buildHazardMarkers();


    runFlowSimulation();


    const seismicEvents =
      Number(
        prediction
          .seismic_events_used ||
        0
      );


    toast(
      `Live data refreshed · ${seismicEvents} seismic events considered.`,
      'success',
      4000
    );


  } catch (
    error
  ) {

    console.error(
      '3D live refresh failed:',
      error
    );


    toast(
      'Could not refresh live hazard data — check the backend.',
      'danger'
    );

  }

}


/* ============================================================
   LIVE 3D AUTO-REFRESH
   ============================================================ */

let live3DRefreshTimer =
  null;


function start3DLiveRefresh() {

  if (
    live3DRefreshTimer
  ) {

    clearInterval(
      live3DRefreshTimer
    );

  }


  live3DRefreshTimer =
    setInterval(
      refreshLiveDataOnly,
      10 *
      60 *
      1000
    );

}


/* ============================================================
   UI WIRING
   ============================================================ */

function wireUI() {

  const baseLayerSel =
    document.getElementById(
      'baseLayerSelect'
    );


  if (
    baseLayerSel
  ) {

    baseLayerSel.value =
      state.baseLayer;


    baseLayerSel.addEventListener(
      'change',
      e =>
        setBaseLayer(
          e.target.value
        )
    );

  }


  const districtSel =
    document.getElementById(
      'districtFocus'
    );


  DISTRICTS.forEach(
    d => {

      const opt =
        document.createElement(
          'option'
        );


      opt.value =
        d.id;


      opt.textContent =
        `${d.name} (${d.state})`;


      districtSel.appendChild(
        opt
      );

    }
  );


  districtSel.addEventListener(
    'change',
    e => {

      if (
        e.target.value ===
        'all'
      ) {

        setDefaultCameraPosition();


        state.controls.target.set(
          0,
          0,
          0
        );


        return;

      }


      const d =
        DISTRICTS.find(
          x =>
            x.id ===
            e.target.value
        );


      if (
        !d
      ) {

        return;

      }


      const {
        x,
        z
      } =
        latLonToXZ(
          d.lat,
          d.lon
        );


      const y =
        terrainHeightAtLatLon(
          d.lat,
          d.lon
        );


      state.controls.target.set(
        x,
        y,
        z
      );


      state.camera.position.set(
        x + 60,
        y + 70,
        z + 60
      );

    }
  );


  const exagSlider =
    document.getElementById(
      'exagSlider'
    );


  exagSlider.addEventListener(
    'input',
    e => {

      state.exaggeration =
        parseFloat(
          e.target.value
        );


      document.getElementById(
        'exagOut'
      ).textContent =
        state.exaggeration.toFixed(
          1
        ) +
        'x';


      if (
        state.grid.length
      ) {

        buildTerrainMesh();

        buildHazardMarkers();

        runFlowSimulation();

      }

    }
  );


  const blendSlider =
    document.getElementById(
      'blendSlider'
    );


  blendSlider.addEventListener(
    'input',
    e => {

      state.hazardBlend =
        parseInt(
          e.target.value,
          10
        ) /
        100;


      document.getElementById(
        'blendOut'
      ).textContent =
        e.target.value +
        '%';


      if (
        state.grid.length
      ) {

        buildTerrainMesh();

      }

    }
  );


  const markerThreshSlider =
    document.getElementById(
      'markerThreshSlider'
    );


  markerThreshSlider.addEventListener(
    'input',
    e => {

      state.markerThreshold =
        parseInt(
          e.target.value,
          10
        );


      document.getElementById(
        'markerThreshOut'
      ).textContent =
        e.target.value +
        '%';


      if (
        state.grid.length
      ) {

        buildHazardMarkers();

      }

    }
  );


  const initThreshSlider =
    document.getElementById(
      'initThreshSlider'
    );


  initThreshSlider.addEventListener(
    'input',
    e => {

      state.initiationThreshold =
        parseInt(
          e.target.value,
          10
        );


      document.getElementById(
        'initThreshOut'
      ).textContent =
        e.target.value +
        '%';

    }
  );


  initThreshSlider.addEventListener(
    'change',
    () => {

      if (
        state.grid.length
      ) {

        runFlowSimulation();

      }

    }
  );


  const agentsSlider =
    document.getElementById(
      'agentsSlider'
    );


  agentsSlider.addEventListener(
    'input',
    e => {

      state.agentsPerPoint =
        parseInt(
          e.target.value,
          10
        );


      document.getElementById(
        'agentsOut'
      ).textContent =
        e.target.value;

    }
  );


  agentsSlider.addEventListener(
    'change',
    () => {

      if (
        state.grid.length
      ) {

        runFlowSimulation();

      }

    }
  );


  document.getElementById(
    'toggleMarkers'
  ).addEventListener(
    'change',
    e => {

      state.markerGroup.visible =
        e.target.checked;

    }
  );


  document.getElementById(
    'toggleFlow'
  ).addEventListener(
    'change',
    () => {

      if (
        state.grid.length
      ) {

        runFlowSimulation();

      }

    }
  );


  document.getElementById(
    'toggleWireframe'
  ).addEventListener(
    'change',
    e => {

      state.wireframe =
        e.target.checked;


      if (
        state.terrainMesh
      ) {

        state.terrainMesh
          .material
          .wireframe =
          state.wireframe;

      }

    }
  );


  document.getElementById(
    'toggleAutoRotate'
  ).addEventListener(
    'change',
    e => {

      state.controls.autoRotate =
        e.target.checked;

    }
  );


  document.getElementById(
    'rerunSimBtn'
  ).addEventListener(
    'click',
    () => {

      if (
        state.grid.length
      ) {

        runFlowSimulation();

      }

    }
  );


  document.getElementById(
    'refreshDataBtn'
  ).addEventListener(
    'click',
    refreshLiveDataOnly
  );


  document.getElementById(
    'resetCameraBtn'
  ).addEventListener(
    'click',
    () => {

      setDefaultCameraPosition();


      state.controls.target.set(
        0,
        0,
        0
      );


      districtSel.value =
        'all';

    }
  );


  document.getElementById(
    'panelToggle'
  ).addEventListener(
    'click',
    () => {

      const panel =
        document.getElementById(
          'controlPanel'
        );


      panel.classList.toggle(
        'hidden'
      );

    }
  );


  document.getElementById(
    'gridNLabel'
  ).textContent =
    `${GRID_N}×${GRID_N}`;

}


function tickClock() {

  const el =
    document.getElementById(
      'clock'
    );


  if (
    el
  ) {

    el.textContent =
      new Date().toLocaleString(
        'en-IN',
        {

          weekday:
            'short',

          day:
            '2-digit',

          month:
            'short',

          year:
            'numeric',

          hour:
            '2-digit',

          minute:
            '2-digit',

          second:
            '2-digit',

          hour12:
            false

        }
      );

  }

}


setInterval(
  tickClock,
  1000
);


tickClock();


/* ---------------- BOOT ---------------- */

wireUI();


initScene();


loadAndBuildAll();


/* Start automatic live-data refresh. */

start3DLiveRefresh();