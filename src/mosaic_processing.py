# src/mosaic_processing.py
"""
Utilities to build weekly cloud-masked mosaics from Sentinel-2 Surface Reflectance
and optionally export the results to Google Drive.

Main features
-------------
- Cloud masking via MSK_CLDPRB (cloud probability) and SCL (scene class).
- Weekly composites using "median", "mean", or "mosaic" reducers.
- Two common products:
    • RGB composites (B4/B3/B2)
    • NDVI composites (band name: "NDVI")
- Convenience exporters for single images and whole weekly collections.

Quick example
-------------
Example:
    import os, ee
    from mosaic_processing import MosaicProcessing, export_all_weeks

    ee.Authenticate()
    ee.Initialize(project=os.getenv("EE_PROJECT_ID"))

    aoi = ee.Geometry.Rectangle([-104.0, 36.0, -80.0, 49.0])

    mp = MosaicProcessing(
        aoi=aoi,
        start_date="2021-04-01",
        end_date="2021-10-01",
        cloud_prob_max=40,
        bands=["B2","B3","B4","B8","B11","B12","SCL","MSK_CLDPRB"]
    )

    rgb = mp.process_rgb(reducer="median")
    ndvi = mp.process_ndvi(reducer="median")

    export_all_weeks(rgb,  folder="rgb_2021",  scale=10, max_images=4)
    export_all_weeks(ndvi, folder="ndvi_2021", scale=20, max_images=4)
"""

from __future__ import annotations
from typing import Optional, List, Literal
from datetime import datetime
import ee


class MosaicProcessing:
    """
    Build weekly cloud-masked mosaics from Sentinel-2 SR for a given AOI/date range.

    Methods
    -------
    process_rgb(...)
        Return an ee.ImageCollection of weekly composites with bands [B4, B3, B2].
    process_ndvi(...)
        Return an ee.ImageCollection of weekly NDVI composites (band "NDVI").

    Parameters
    ----------
    aoi : ee.Geometry
        Area of Interest. Any EE geometry is supported (Polygon, Rectangle, etc.).
    start_date : str
        Start date in ISO format "YYYY-MM-DD".
    end_date : str
        End date in ISO format "YYYY-MM-DD".
    collection_id : str, default "COPERNICUS/S2_SR"
        Earth Engine dataset ID to use.
    cloud_prob_max : int, default 40
        Maximum value of MSK_CLDPRB (0..100) to keep a pixel.
    bands : list[str] | None, default None
        Optional safe-select list of bands to keep after masking (missing bands ignored).
    scale_to_reflectance : bool, default True
        Kept for API completeness only; Sentinel-2 SR is already scaled.
    path_to_drive : str, default ""
        Reserved for future use.

    Example
    -------
    Example:
        mp = MosaicProcessing(
            aoi=ee.Geometry.Rectangle([-104,36,-80,49]),
            start_date="2022-05-01",
            end_date="2022-09-01",
            cloud_prob_max=35
        )
    """

    def __init__(
        self,
        aoi: ee.Geometry,
        start_date: str,
        end_date: str,
        collection_id: str = "COPERNICUS/S2_SR",
        cloud_prob_max: int = 40,
        bands: Optional[List[str]] = None,   # e.g. ["B2","B3","B4","B8","B11","B12","SCL","MSK_CLDPRB"]
        scale_to_reflectance: bool = True,   # kept for API completeness; S2_SR is already scaled
        path_to_drive: str = ""
    ):
        """
        Initialize the processor and pre-build a masked Sentinel-2 collection.

        Raises
        ------
        ee.EEException
            If the Earth Engine client is not initialized.
        ValueError
            If dates are malformed.
        """
        self.aoi = aoi
        self.start = ee.Date(start_date)
        self.end = ee.Date(end_date)
        self.collection_id = collection_id
        self.cloud_prob_max = cloud_prob_max
        self.scale_to_reflectance = scale_to_reflectance
        self.bands = bands
        self.col = self._build_collection()

    # -------------------- public API --------------------

    def process_rgb(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        reducer: Literal["median","mean","mosaic"] = "median"
    ) -> ee.ImageCollection:
        """
        Create weekly natural-color composites with bands [B4, B3, B2].

        Parameters
        ----------
        start : str | None, default None
            Optional override of start date (ISO "YYYY-MM-DD") for this call only.
        end : str | None, default None
            Optional override of end date (ISO "YYYY-MM-DD") for this call only.
        reducer : {"median","mean","mosaic"}, default "median"
            Spatial reducer within each week:
              - "median": robust per-pixel median
              - "mean":   per-pixel mean
              - "mosaic": last-on-top mosaic (fast visual composite)

        Returns
        -------
        ee.ImageCollection
            Weekly composites clipped to `aoi`, each image holding [B4, B3, B2].
            Properties include: "system:time_start", "week_start", "week_end", "reducer".

        Example
        -------
        Example:
            rgb = mp.process_rgb(reducer="median")
            first = ee.Image(rgb.first())
            print(first.bandNames().getInfo())  # ['B4','B3','B2']
        """
        def keep_rgb(i): return i.select(["B4","B3","B2"])
        col = self.col.map(keep_rgb)
        return self._weekly_composite(col, reducer=reducer, start=start, end=end)

    def process_ndvi(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        reducer: Literal["median","mean","mosaic"] = "median"
    ) -> ee.ImageCollection:
        """
        Create weekly NDVI composites (band "NDVI").
        NDVI = (B8 - B4) / (B8 + B4), computed per image then composited weekly.

        Parameters
        ----------
        start : str | None, default None
            Optional override of start date (ISO "YYYY-MM-DD") for this call only.
        end : str | None, default None
            Optional override of end date (ISO "YYYY-MM-DD") for this call only.
        reducer : {"median","mean","mosaic"}, default "median"
            Reducer for compositing the per-image NDVI within each week.

        Returns
        -------
        ee.ImageCollection
            Weekly NDVI composites (single band "NDVI"), clipped to `aoi`.
            Properties include: "system:time_start", "week_start", "week_end", "reducer".

        Example
        -------
        Example:
            ndvi = mp.process_ndvi(reducer="mean")
            img = ee.Image(ndvi.first())
            print(img.bandNames().getInfo())  # ['NDVI']
        """
        def add_ndvi(i):
            ndvi = i.normalizedDifference(["B8","B4"]).rename("NDVI")
            return i.addBands(ndvi).select(["NDVI"])
        col = self.col.map(add_ndvi)
        return self._weekly_composite(col, reducer=reducer, start=start, end=end)

    # ------------------- internals ---------------------

    def _build_collection(self) -> ee.ImageCollection:
        """
        Internal: build the base masked collection.

        Returns
        -------
        ee.ImageCollection
            Sentinel-2 SR images filtered by AOI/date and cloud-masked.

        Notes
        -----
        - If `bands` were specified in the constructor, they are safely selected:
          bands not present in a given image are ignored, preventing select errors.
        - The original "system:time_start" property is preserved on each image.

        Example (debug/inspection only)
        -------------------------------
        Example:
            # Not typically called directly; for inspection you might do:
            base = mp._build_collection()
            print(base.first().propertyNames().getInfo())
        """
        col = (ee.ImageCollection(self.collection_id)
               .filterBounds(self.aoi)
               .filterDate(self.start, self.end))

        def prep(i: ee.Image) -> ee.Image:
            i = self._mask_s2(i)
            if self.bands:
                # keep only bands that exist to avoid select errors
                existing = i.bandNames()
                wanted  = ee.List(self.bands)
                keep    = wanted.filter(ee.Filter.inList("item", existing))
                i = i.select(keep)
            return i.copyProperties(i, ["system:time_start"])

        return col.map(prep)

    def _mask_s2(self, img: ee.Image) -> ee.Image:
        """
        Internal: apply Sentinel-2 cloud & class masks.

        Masking rules
        -------------
        1) If present, keep pixels where MSK_CLDPRB < cloud_prob_max.
        2) Always apply SCL mask to keep useful classes:
           vegetation(4–6), bare(7), snow(10), water(11).

        Parameters
        ----------
        img : ee.Image
            A Sentinel-2 SR image.

        Returns
        -------
        ee.Image
            Masked image with the same bands as input.

        Example (debug only)
        --------------------
        Example:
            # For a single image, you could inspect the mask like this:
            i = ee.Image(mp._build_collection().first())
            masked = mp._mask_s2(i)
        """
        # Cloud prob (0-100): keep where < threshold, if band exists
        def with_cp(ii): return ii.updateMask(ii.select("MSK_CLDPRB").lt(self.cloud_prob_max))
        img = ee.Image(ee.Algorithms.If(img.bandNames().contains("MSK_CLDPRB"), with_cp(img), img))
        # SCL: keep vegetation(4–6), bare(7), snow(10), water(11); drop clouds/shadows
        scl  = img.select("SCL")
        good = (scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
                .Or(scl.eq(7)).Or(scl.eq(10)).Or(scl.eq(11)))
        return img.updateMask(good)

    def _weekly_composite(
        self,
        col: ee.ImageCollection,
        reducer: Literal["median","mean","mosaic"] = "median",
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> ee.ImageCollection:
        """
        Internal: convert a (masked) collection into weekly composites.

        Parameters
        ----------
        col : ee.ImageCollection
            Pre-masked collection to composite weekly.
        reducer : {"median","mean","mosaic"}, default "median"
            Spatial reducer to apply within each 7-day window.
        start : str | None, default None
            Optional override of start date in ISO format.
        end : str | None, default None
            Optional override of end date in ISO format. If None, defaults to *today*.

        Returns
        -------
        ee.ImageCollection
            Weekly composites clipped to `aoi`. Each image carries:
            - "system:time_start" (epoch ms)
            - "week_start", "week_end" (ISO strings)
            - "reducer" (the reducer used)

        Example (advanced)
        ------------------
        Example:
            # Not needed in normal use; called internally by process_* methods.
            weekly = mp._weekly_composite(mp._build_collection(), reducer="median")
            print(weekly.size().getInfo())
        """
        # dates: default to instance range; end defaults to today if omitted by caller
        s = ee.Date(start) if start else self.start
        e = ee.Date(end)   if end   else (ee.Date(datetime.today().strftime("%Y-%m-%d")) if end is None else self.end)

        red = {"median": ee.Reducer.median(), "mean": ee.Reducer.mean(), "mosaic": None}[reducer]
        ticks = ee.List.sequence(s.millis(), e.millis(), 7 * 24 * 3600 * 1000)

        def per_week(ms):
            ws = ee.Date(ms)
            we = ws.advance(7, "day")
            wk = col.filterDate(ws, we)
            img = wk.mosaic() if reducer == "mosaic" else wk.reduce(red)
            img = ee.Image(img).clip(self.aoi).set({
                "system:time_start": ws.millis(),
                "week_start": ws.format("YYYY-MM-dd"),
                "week_end": we.format("YYYY-MM-dd"),
                "reducer": reducer
            })
            return img

        return ee.ImageCollection(ticks.map(per_week))


# ------------------- export helpers ---------------------

def export_to_drive(
    image: ee.Image,
    description: str,
    folder: str = "ee-exports",
    region: ee.Geometry | None = None,
    scale: int = 20,
    crs: str = "EPSG:4326",
    file_format: str = "GeoTIFF",
    max_pixels: int = 1_000_000_000
) -> ee.batch.Task:
    """
    Export a single Earth Engine image to Google Drive.

    Parameters
    ----------
    image : ee.Image
        Image to export (e.g., one weekly composite).
    description : str
        Task name and file prefix in Drive.
    folder : str, default "ee-exports"
        Drive folder name. Auto-created if it doesn't exist.
    region : ee.Geometry | None, default None
        Export geometry. Defaults to image geometry if not provided.
    scale : int, default 20
        Pixel size in meters (S2 native is 10 m; 20 m is common for NDVI).
    crs : str, default "EPSG:4326"
        Output CRS. Choose a projected CRS for area-accurate analytics.
    file_format : str, default "GeoTIFF"
        Output format (e.g., "GeoTIFF", "TFRecord").
    max_pixels : int, default 1_000_000_000
        Maximum pixels allowed for the export.

    Returns
    -------
    ee.batch.Task
        The started export task.

    Example
    -------
    Example:
        ee.Initialize(project=os.getenv("EE_PROJECT_ID"))
        aoi = ee.Geometry.Rectangle([-104,36,-80,49])
        mp = MosaicProcessing(aoi, "2021-04-01", "2021-10-01")
        img = ee.Image(mp.process_ndvi().first())
        task = export_to_drive(img, description="ndvi_2021_week1",
                               folder="ndvi_2021", scale=20)
        # Monitor in EE Code Editor → Tasks
    """
    if region is None:
        region = image.geometry()

    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=description,
        region=region,
        scale=scale,
        crs=crs,
        fileFormat=file_format,
        maxPixels=max_pixels
    )
    task.start()
    print(f"🚀 Export started: {description}")
    print(f"📂 Folder: {folder} | Scale: {scale} m | CRS: {crs}")
    return task


def export_all_weeks(
    imgcol: ee.ImageCollection,
    folder: str = "weekly_rgb",
    scale: int = 20,
    max_images: Optional[int] = None
) -> None:
    """
    Export each image in a weekly ImageCollection to Google Drive.

    Parameters
    ----------
    imgcol : ee.ImageCollection
        Typically the output of process_rgb() or process_ndvi().
    folder : str, default "weekly_rgb"
        Drive folder name where images will be saved.
    scale : int, default 20
        Pixel size in meters for all exports.
    max_images : int | None, default None
        If provided, only the first `max_images` images are exported (helpful for quotas).

    Returns
    -------
    None
        Starts one EE export task per image (beware of task limits).

    Notes
    -----
    - Earth Engine enforces a limit on simultaneous tasks (often ~100).
      Use `max_images`, export in batches, or monitor from the EE Code Editor.
    - Each file will be named with prefix "{folder}_{week_start}".

    Example
    -------
    Example:
        ee.Initialize(project=os.getenv("EE_PROJECT_ID"))
        aoi = ee.Geometry.Rectangle([-104,36,-80,49])
        mp = MosaicProcessing(aoi, "2022-06-01", "2022-09-01")

        rgb = mp.process_rgb(reducer="median")
        export_all_weeks(rgb, folder="rgb_weeks_2022", scale=10, max_images=6)

        ndvi = mp.process_ndvi(reducer="median")
        export_all_weeks(ndvi, folder="ndvi_weeks_2022", scale=20, max_images=6)
    """
    n = imgcol.size().getInfo()
    if max_images:
        n = min(n, max_images)

    img_list = imgcol.toList(n)
    for i in range(n):
        img = ee.Image(img_list.get(i))
        # Safer date extraction
        date = img.get("week_start").getInfo() or img.date().format("YYYY-MM-dd").getInfo()
        desc = f"{folder}_{date}"
        export_to_drive(
            image=img,
            description=desc,
            folder=folder,
            scale=scale
        )
    print(f"Started {n} export tasks to Google Drive ({folder}).")
