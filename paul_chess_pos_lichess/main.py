from img2table.document import Image
from img2table.ocr import TesseractOCR
from PIL import Image as PILImage
import cv2

tesseract_ocr = TesseractOCR(n_threads=1, lang="eng")


# Instantiation of the image
img = Image(src="paul_chess_pos_lichess/img/IMG_2022.JPG")
print("ing")

# Table identification
extracted_tables = img.extract_tables(ocr=tesseract_ocr)

table_img = cv2.imread("paul_chess_pos_lichess/img/IMG_2022.JPG")
print("read")

for table in extracted_tables:
    for row in table.content.values():
        for cell in row:
            cv2.rectangle(
                table_img,
                (cell.bbox.x1, cell.bbox.y1),
                (cell.bbox.x2, cell.bbox.y2),
                (255, 0, 0),
                2,
            )

PILImage.fromarray(table_img).save("test.png")


print(extracted_tables)
