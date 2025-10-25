# src/mosaic_processing.py
from __future__ import annotations
from typing import Optional, List, Literal
from datetime import datetime
import ee

class MosaicProcessing:
    """
    Weekly cloud-masked mosaics from Sentinel-2 SR:
      • process_rgb()  -> weekly B4/B3/B2 composites
      • process_ndvi() -> weekly NDVI composites
    """

    def __init__(
        self,
        aoi: ee.Geometry,
        start_date: str,
        end_date: str,
        collection_id: str = "COPERNICUS/S2_SR",
        cloud_prob_max: int = 40,
        bands: Optional[List[str]] = None,   # e.g. ["B2","B3","B4","B8","B11","B12","SCL","MSK_CLDPRB"]
        scale_to_reflectance: bool = True,    # kept for API completeness; S2_SR is already scaled
        path_to_drive: str = ""
    ):
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
        """Weekly composites with natural-color bands [B4,B3,B2]."""
        def keep_rgb(i): return i.select(["B4","B3","B2"])
        col = self.col.map(keep_rgb)
        return self._weekly_composite(col, reducer=reducer, start=start, end=end)

    def process_ndvi(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None,
        reducer: Literal["median","mean","mosaic"] = "median"
    ) -> ee.ImageCollection:
        """Weekly composites of NDVI = (B8 - B4) / (B8 + B4)."""
        def add_ndvi(i):
            ndvi = i.normalizedDifference(["B8","B4"]).rename("NDVI")
            return i.addBands(ndvi).select(["NDVI"])
        col = self.col.map(add_ndvi)
        return self._weekly_composite(col, reducer=reducer, start=start, end=end)

    # def process_sdvi(self):

    # ------------------- internals ---------------------

    def _build_collection(self) -> ee.ImageCollection:
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

    # ------------------- export ---------------------
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
    Exports a single Earth Engine image to Google Drive.

    Parameters
    ----------
    image : ee.Image
        The image to export.
    description : str
        Task name and file prefix in Drive.
    folder : str, default 'ee-exports'
        Drive folder where the file will be saved.
    region : ee.Geometry or None
        Export region (defaults to image bounds if None).
    scale : int, default 20
        Pixel resolution in meters.
    crs : str, default 'EPSG:4326'
        Coordinate reference system.
    file_format : str, default 'GeoTIFF'
        Output file format ('GeoTIFF' or 'TFRecord', etc.).
    max_pixels : int, default 1e9
        Maximum pixel count allowed by EE.
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


def export_all_weeks(imgcol, folder="weekly_rgb", scale=20, max_images=None):
    """
    Exports each weekly image in an ImageCollection to Google Drive.
    Works for any product (RGB, NDVI, etc.).
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
