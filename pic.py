from PIL import Image, ImageDraw, ImageFont
import os, math

OUT = os.getcwd()
os.makedirs(OUT, exist_ok=True)

# ── Colors ──────────────────────────────────────────
BG       = (245, 248, 244)
GREEN    = (52, 120, 68)
LGREEN   = (134, 191, 110)
DGREEN   = (28, 72, 40)
WHITE    = (255, 255, 255)
LGRAY    = (220, 230, 220)
DGRAY    = (80, 90, 80)
ACCENT   = (255, 183, 3)
BLUE     = (52, 100, 180)
LBLUE    = (180, 210, 255)
RED      = (200, 60, 60)
ORANGE   = (220, 130, 30)

def font(size):
    for name in ["/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
                 "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf"]:
        try: return ImageFont.truetype(name, size)
        except: pass
    return ImageFont.load_default()

def fontR(size):
    for name in ["/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                 "/usr/share/fonts/truetype/freefont/FreeSans.ttf"]:
        try: return ImageFont.truetype(name, size)
        except: pass
    return ImageFont.load_default()

def new_canvas(w=1200, h=800):
    img = Image.new("RGB", (w, h), BG)
    d   = ImageDraw.Draw(img)
    # subtle grid
    for x in range(0, w, 40):
        d.line([(x,0),(x,h)], fill=(230,238,230), width=1)
    for y in range(0, h, 40):
        d.line([(0,y),(w,y)], fill=(230,238,230), width=1)
    return img, d

def header(d, w, title, subtitle, num):
    d.rectangle([(0,0),(w,70)], fill=GREEN)
    d.text((20,18), f"Figure {num}", font=font(18), fill=ACCENT)
    d.text((130,18), title, font=font(20), fill=WHITE)
    if subtitle:
        d.text((20,48), subtitle, font=fontR(13), fill=LGRAY)

def box(d, x1,y1,x2,y2, fill=WHITE, outline=GREEN, radius=10, lw=2):
    d.rounded_rectangle([(x1,y1),(x2,y2)], radius=radius, fill=fill, outline=outline, width=lw)

def arrow(d, x1,y1,x2,y2, color=GREEN, w=2):
    d.line([(x1,y1),(x2,y2)], fill=color, width=w)
    # arrowhead
    angle = math.atan2(y2-y1, x2-x1)
    size  = 10
    d.polygon([
        (x2, y2),
        (int(x2 - size*math.cos(angle-0.4)), int(y2 - size*math.sin(angle-0.4))),
        (int(x2 - size*math.cos(angle+0.4)), int(y2 - size*math.sin(angle+0.4))),
    ], fill=color)

def label(d, x, y, text, fnt=None, fill=DGRAY, anchor="mm"):
    if fnt is None: fnt = fontR(14)
    d.text((x,y), text, font=fnt, fill=fill, anchor=anchor)

def save(img, num, name):
    path = f"{OUT}/Fig{num:02d}_{name}.png"
    img.save(path, "PNG", dpi=(150,150))
    print(f"  Saved: {path}")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3.1 – System Architecture
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
header(d, 1200, "High-Level System Architecture of GreenLeaf", "Three-tier client-server pattern with external API services", "3.1")

# Tier labels on left
for (y, txt, clr) in [(150,"PRESENTATION TIER",LGREEN),(370,"APPLICATION TIER",GREEN),(600,"EXTERNAL API TIER",DGREEN)]:
    d.rectangle([(10,y-20),(120,y+80)], fill=clr)
    for i,ch in enumerate(txt.split()):
        d.text((18,y-10+i*16), ch, font=fontR(12), fill=WHITE)

# User box
box(d,160,130,360,210, fill=LBLUE, outline=BLUE)
d.text((195,158), "FARMER / USER", font=font(14), fill=BLUE)
d.text((175,180), "Phone/Desktop Browser", font=fontR(12), fill=DGRAY)

# Streamlit UI
box(d,430,130,750,210, fill=WHITE, outline=GREEN)
d.text((490,148), "Streamlit Web Interface", font=font(15), fill=GREEN)
d.text((445,172), "File Uploader  |  City Input  |  Results Display", font=fontR(12), fill=DGRAY)

# Arrow user → UI
arrow(d, 362,170, 428,170)

# Python Backend box
box(d,230,330,970,430, fill=WHITE, outline=GREEN)
d.text((540,345), "Python Application Backend", font=font(16), fill=GREEN)

# Sub-modules inside backend
for (x,lbl) in [(250,"Image\nPreprocessor"),(420,"Parallel API\nDispatcher"),(590,"Symptom\nScanner"),(760,"Report\nSynthesizer"),(900,"Weather\nFetcher")]:
    box(d,x,365,x+130,420, fill=LGRAY, outline=GREEN, radius=6)
    for i,ln in enumerate(lbl.split("\n")):
        d.text((x+65, 375+i*16), ln, font=fontR(12), fill=DGREEN, anchor="mm")

# Arrow UI → backend
arrow(d, 590,212, 590,328)

# External APIs
apis = [
    (180, "PlantNet API v2", "Species ID\n+ Disease", LGREEN, GREEN),
    (490, "Google Gemini\n1.5 Flash", "Natural Language\nReport", (200,230,255), BLUE),
    (800, "OpenWeatherMap\nAPI", "Temperature\n& Humidity", (255,240,200), ORANGE),
]
for (x,title,sub,bg,fg) in apis:
    box(d,x,570,x+280,680, fill=bg, outline=fg, radius=12)
    for i,ln in enumerate(title.split("\n")):
        d.text((x+140,585+i*18), ln, font=font(15), fill=fg, anchor="mm")
    for i,ln in enumerate(sub.split("\n")):
        d.text((x+140,625+i*15), ln, font=fontR(13), fill=DGRAY, anchor="mm")
    arrow(d, x+140,432, x+140,568, color=fg)

# Caption
d.text((600,750), "Figure 3.1 – High-Level System Architecture of GreenLeaf", font=fontR(15), fill=DGRAY, anchor="mm")
save(img, 31, "System_Architecture")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3.2 – Use Case Diagram
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
header(d, 1200, "Use Case Diagram – GreenLeaf Application", "Actors and primary system interactions", "3.2")

# System boundary
d.rectangle([(280,100),(920,720)], outline=GREEN, width=3)
d.text((560,112), "«system» GreenLeaf", font=font(15), fill=GREEN)

# Actor: Farmer
d.ellipse([(60,240),(120,300)], outline=BLUE, width=3)          # head
d.line([(90,300),(90,390)], fill=BLUE, width=3)                 # body
d.line([(90,330),(50,370)], fill=BLUE, width=3)                 # left arm
d.line([(90,330),(130,370)], fill=BLUE, width=3)                # right arm
d.line([(90,390),(60,440)], fill=BLUE, width=3)                 # left leg
d.line([(90,390),(120,440)], fill=BLUE, width=3)                # right leg
d.text((90,450), "Farmer", font=font(14), fill=BLUE, anchor="mm")

# Actor: Admin
d.ellipse([(1080,240),(1140,300)], outline=RED, width=3)
d.line([(1110,300),(1110,390)], fill=RED, width=3)
d.line([(1110,330),(1070,370)], fill=RED, width=3)
d.line([(1110,330),(1150,370)], fill=RED, width=3)
d.line([(1110,390),(1080,440)], fill=RED, width=3)
d.line([(1110,390),(1140,440)], fill=RED, width=3)
d.text((1110,450), "Admin", font=font(14), fill=RED, anchor="mm")

# Use cases
farmer_uc = [
    (580,180,"Upload Leaf Image"),
    (580,260,"Enter City Name"),
    (580,340,"Trigger AI Analysis"),
    (580,420,"View Plant ID Result"),
    (580,500,"View Disease Result"),
    (580,580,"Read Treatment Report"),
    (580,660,"View Prevention Tips"),
]
admin_uc = [
    (750,300,"Update API Keys"),
    (750,420,"Monitor App Logs"),
    (750,540,"Deploy Updates"),
]
for (x,y,txt) in farmer_uc:
    d.ellipse([(x-140,y-22),(x+140,y+22)], fill=WHITE, outline=GREEN, width=2)
    d.text((x,y), txt, font=fontR(13), fill=DGREEN, anchor="mm")
    d.line([(120,320),(x-140,y)], fill=BLUE, width=1)

for (x,y,txt) in admin_uc:
    d.ellipse([(x-110,y-22),(x+110,y+22)], fill=WHITE, outline=RED, width=2)
    d.text((x,y), txt, font=fontR(13), fill=RED, anchor="mm")
    d.line([(1080,320),(x+110,y)], fill=RED, width=1)

d.text((600,760), "Figure 3.2 – Use Case Diagram – GreenLeaf Application", font=fontR(15), fill=DGRAY, anchor="mm")
save(img, 32, "Use_Case_Diagram")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 3.3 – Level 1 DFD
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
header(d, 1200, "Level 1 Data Flow Diagram – GreenLeaf Processing Pipeline", "Four-process data transformation from input to diagnosis report", "3.3")

# External entities (rectangles)
def entity(d, x,y, txt):
    d.rectangle([(x,y),(x+160,y+60)], fill=LGRAY, outline=DGREEN, width=2)
    d.text((x+80,y+30), txt, font=font(13), fill=DGREEN, anchor="mm")

entity(d,  40, 370, "Farmer")
entity(d,1000, 370, "Report")

# Processes (circles)
procs = [
    (280, 380, "P1\nImage\nPreprocess"),
    (480, 250, "P2\nPlantNet\nIdentify"),
    (480, 510, "P3\nPlantNet\nDisease"),
    (700, 380, "P4\nGemini\nReport Gen"),
    (860, 220, "P5\nWeather\nFetch"),
]
for (x,y,txt) in procs:
    d.ellipse([(x-70,y-55),(x+70,y+55)], fill=WHITE, outline=GREEN, width=2)
    for i,ln in enumerate(txt.split("\n")):
        d.text((x,y-28+i*20), ln, font=fontR(13), fill=GREEN, anchor="mm")

# Data stores (open rectangles)
def store(d, x,y,txt):
    d.rectangle([(x,y),(x+180,y+40)], fill=(240,255,240), outline=GREEN, width=2)
    d.line([(x,y),(x,y+40)], fill=BG, width=4)   # open left
    d.text((x+90,y+20), txt, font=fontR(12), fill=DGREEN, anchor="mm")

store(d, 600, 140, "D1: Image Data")
store(d, 600, 660, "D2: API Results")

# Arrows
flows = [
    (200,400,  210,400, "Leaf Image"),
    (352,365,  412,295, "Image Bytes"),
    (352,395,  412,520, "Image Bytes"),
    (550,270,  630,370, "Species + Score"),
    (550,510,  630,410, "Disease + Score"),
    (770,380,  998,390, "Diagnosis Report"),
    (860,275,  780,365, "Temp + Humidity"),
]
for (x1,y1,x2,y2,lbl) in flows:
    arrow(d,x1,y1,x2,y2)
    mx,my = (x1+x2)//2,(y1+y2)//2
    d.text((mx+5,my-14), lbl, font=fontR(11), fill=ORANGE)

d.text((600,770), "Figure 3.3 – Level 1 Data Flow Diagram – GreenLeaf Processing Pipeline", font=fontR(15), fill=DGRAY, anchor="mm")
save(img, 33, "DFD_Level1")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4.1 – Homepage Hero
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
# Dark hero background
d.rectangle([(0,0),(1200,820)], fill=(18,38,20))
# Decorative circles
for (cx,cy,r,a) in [(1100,100,180,(30,80,35)),(80,750,140,(25,65,30))]:
    d.ellipse([(cx-r,cy-r),(cx+r,cy+r)], fill=a)
# GreenLeaf branding
d.text((600,120), "🌿 GreenLeaf", font=font(52), fill=LGREEN, anchor="mm")
d.text((600,185), "AI-Powered Plant Disease Detection", font=fontR(24), fill=LGRAY, anchor="mm")
d.line([(400,210),(800,210)], fill=GREEN, width=2)

# Step indicators
steps = ["1\nUpload\nLeaf Photo", "2\nEnter\nCity Name", "3\nClick\nAnalyze", "4\nRead\nDiagnosis"]
colors= [LGREEN, ACCENT, ORANGE, RED]
for i,(st,cl) in enumerate(zip(steps, colors)):
    sx = 230 + i*195
    d.ellipse([(sx-40,300),(sx+40,380)], fill=cl)
    d.text((sx,340), st.split("\n")[0], font=font(22), fill=WHITE, anchor="mm")
    if i<3: arrow(d, sx+42,340, sx+153,340, color=LGRAY, w=2)
    for j,ln in enumerate(st.split("\n")[1:]):
        d.text((sx,400+j*22), ln, font=fontR(14), fill=LGRAY, anchor="mm")

# Feature pills
pills = ["Species Identification","Disease Detection","AI Report","Weather Context","Fallback Scanner"]
for i,p in enumerate(pills):
    px = 130+i*196
    d.rounded_rectangle([(px,490),(px+175,530)], radius=20, fill=(30,70,35), outline=LGREEN, width=1)
    d.text((px+87,510), p, font=fontR(13), fill=LGREEN, anchor="mm")

# Bottom tagline
d.text((600,610), "No installation • No registration • Works on any device", font=fontR(18), fill=LGRAY, anchor="mm")
d.rectangle([(350,650),(850,710)], fill=GREEN)
d.text((600,680), "Get Started — Upload a Leaf Photo", font=font(18), fill=WHITE, anchor="mm")

d.text((600,770), "Figure 4.1 – GreenLeaf Homepage – Hero Banner with Step-by-Step Workflow Indicator", font=fontR(14), fill=LGRAY, anchor="mm")
save(img, 41, "Homepage_Hero")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4.2 – Upload Panel with Weather Badges
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
d.rectangle([(0,0),(1200,820)], fill=(20,40,22))

# Left panel
d.rounded_rectangle([(30,90),(440,750)], radius=14, fill=(28,55,30), outline=GREEN, width=2)
d.text((235,120), "Upload Leaf Image", font=font(18), fill=LGREEN, anchor="mm")
# Upload zone
d.rounded_rectangle([(60,145),(410,360)], radius=10, fill=(22,44,24), outline=LGREEN, width=2)
d.text((235,220), "📷", font=font(40), fill=LGREEN, anchor="mm")
d.text((235,280), "Drag & drop or click to browse", font=fontR(14), fill=LGRAY, anchor="mm")
d.text((235,305), "JPEG · PNG · WebP · BMP · TIFF", font=fontR(12), fill=(120,160,120), anchor="mm")

d.text((235,390), "City / Location", font=fontR(14), fill=LGRAY, anchor="mm")
d.rounded_rectangle([(60,410),(410,450)], radius=8, fill=(22,44,24), outline=GREEN, width=1)
d.text((80,430), "Jaipur", font=fontR(15), fill=WHITE)

# Weather badges
d.text((235,490), "Live Weather – Jaipur", font=font(14), fill=LGREEN, anchor="mm")
d.rounded_rectangle([(65,515),(195,575)], radius=10, fill=(30,65,35), outline=LGREEN, width=1)
d.text((130,532), "🌡️ Temperature", font=fontR(11), fill=LGRAY, anchor="mm")
d.text((130,556), "34°C", font=font(20), fill=ACCENT, anchor="mm")
d.rounded_rectangle([(215,515),(395,575)], radius=10, fill=(30,65,35), outline=LGREEN, width=1)
d.text((305,532), "💧 Humidity", font=fontR(11), fill=LGRAY, anchor="mm")
d.text((305,556), "58%", font=font(20), fill=BLUE, anchor="mm")

d.rounded_rectangle([(65,590),(395,660)], radius=10, fill=GREEN, outline=LGREEN, width=1)
d.text((230,625), "🔍  Analyze Leaf", font=font(18), fill=WHITE, anchor="mm")

# Right panel – leaf image preview
d.rounded_rectangle([(460,90),(1170,750)], radius=14, fill=(22,44,24), outline=GREEN, width=2)
d.text((815,120), "Image Preview", font=font(16), fill=LGREEN, anchor="mm")
# Simulated leaf image (green gradient blob)
for r in range(200,0,-4):
    alpha = int(60 + 130*(1-r/200))
    d.ellipse([(815-r,340-int(r*0.7)),(815+r,340+int(r*0.7))], fill=(20+r//4, 100+r//3, 30+r//4))
d.text((815,560), "Banana Leaf • Uploaded ✓", font=fontR(14), fill=LGRAY, anchor="mm")
d.text((600,790), "Figure 4.2 – Left Panel – Image Upload Interface with Weather Badges after City Entry", font=fontR(13), fill=LGRAY, anchor="mm")
save(img, 42, "Upload_Panel_Weather")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4.3 – Quick Visual Check
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
d.rectangle([(0,0),(1200,820)], fill=(20,40,22))

d.rounded_rectangle([(100,90),(1100,720)], radius=16, fill=(26,52,28), outline=GREEN, width=2)
d.text((600,130), "Quick Visual Check  —  Pixel-Level Symptom Scanner", font=font(18), fill=LGREEN, anchor="mm")
d.line([(150,155),(1050,155)], fill=GREEN, width=1)

# Gauge circle
cx,cy,R = 600,350,130
for deg in range(0,360,3):
    rad = math.radians(deg)
    r1,r2 = R-12, R
    x1,y1 = cx+int(r1*math.cos(rad)), cy+int(r1*math.sin(rad))
    x2,y2 = cx+int(r2*math.cos(rad)), cy+int(r2*math.sin(rad))
    frac = deg/360
    if frac < 0.33:   c=(80,180,60)
    elif frac < 0.66: c=ACCENT
    else:             c=RED
    d.line([(x1,y1),(x2,y2)], fill=c, width=3)
d.ellipse([(cx-100,cy-100),(cx+100,cy+100)], fill=(26,52,28))
d.text((cx,cy-20), "MODERATE", font=font(18), fill=ACCENT, anchor="mm")
d.text((cx,cy+10), "STRESS", font=font(16), fill=ACCENT, anchor="mm")
d.text((cx,cy+40), "7.3%", font=font(22), fill=WHITE, anchor="mm")
d.text((cx,cy+68), "lesion pixels", font=fontR(13), fill=LGRAY, anchor="mm")

# Stats row
stats = [("Green Pixels","82.1%",LGREEN),("Lesion Pixels","7.3%",ORANGE),("Background","10.6%",LGRAY),("Severity","Moderate",ACCENT)]
for i,(lbl,val,cl) in enumerate(stats):
    sx = 165 + i*230
    d.rounded_rectangle([(sx,530),(sx+200,620)], radius=10, fill=(30,60,32), outline=cl, width=2)
    d.text((sx+100,555), lbl, font=fontR(13), fill=LGRAY, anchor="mm")
    d.text((sx+100,592), val, font=font(18), fill=cl, anchor="mm")

d.text((600,660), "⚠  Visible stress detected. Proceed to full AI analysis for diagnosis.", font=fontR(15), fill=ORANGE, anchor="mm")
d.text((600,780), "Figure 4.3 – Quick Visual Check Panel – Showing Moderate Stress Detection Result", font=fontR(14), fill=LGRAY, anchor="mm")
save(img, 43, "Quick_Visual_Check")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4.4 – Plant Identification Card
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
d.rectangle([(0,0),(1200,820)], fill=(20,40,22))

d.rounded_rectangle([(80,90),(1120,720)], radius=16, fill=(26,52,28), outline=GREEN, width=2)
# Header strip
d.rounded_rectangle([(80,90),(1120,175)], radius=16, fill=GREEN)
d.text((600,132), "🌿  Plant Identification Result", font=font(22), fill=WHITE, anchor="mm")

# Main species name
d.text((600,230), "Banana", font=font(42), fill=LGREEN, anchor="mm")
d.text((600,285), "Musa paradisiaca  L.", font=fontR(20), fill=LGRAY, anchor="mm")
d.line([(200,310),(1000,310)], fill=GREEN, width=1)

# Confidence bar
d.text((600,345), "Identification Confidence", font=fontR(15), fill=LGRAY, anchor="mm")
d.rounded_rectangle([(200,370),(1000,410)], radius=8, fill=(30,60,32))
d.rounded_rectangle([(200,370),(200+int(800*0.87),410)], radius=8, fill=LGREEN)
d.text((1010,390), "87%", font=font(18), fill=ACCENT, anchor="mm")

# Info grid
info = [
    ("Kingdom","Plantae"),("Family","Musaceae"),("Genus","Musa"),("Common Name","Banana"),
    ("Organ Detected","Leaf"),("API Source","PlantNet v2"),
]
for i,(k,v) in enumerate(info):
    col = i%3
    row = i//3
    ix  = 200+col*280
    iy  = 460+row*100
    d.rounded_rectangle([(ix,iy),(ix+260,iy+80)], radius=10, fill=(30,60,32), outline=GREEN, width=1)
    d.text((ix+130, iy+20), k, font=fontR(12), fill=LGRAY, anchor="mm")
    d.text((ix+130, iy+52), v, font=font(15), fill=WHITE, anchor="mm")

d.text((600,700), "✅  Species identified with high confidence (87%)", font=fontR(15), fill=LGREEN, anchor="mm")
d.text((600,780), "Figure 4.4 – Plant Identification Card with Confidence Bar (Banana, Musa paradisiaca, 87%)", font=fontR(13), fill=LGRAY, anchor="mm")
save(img, 44, "Plant_Identification_Card")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4.5 – Disease Detection Card
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
d.rectangle([(0,0),(1200,820)], fill=(20,40,22))

d.rounded_rectangle([(80,90),(1120,720)], radius=16, fill=(26,52,28), outline=ORANGE, width=2)
d.rounded_rectangle([(80,90),(1120,175)], radius=16, fill=ORANGE)
d.text((600,132), "⚠️  Disease Detection Result", font=font(22), fill=WHITE, anchor="mm")

d.text((600,228), "Yellow Sigatoka", font=font(38), fill=ACCENT, anchor="mm")
d.text((600,282), "Banana Sigatoka Disease  •  Pathogen: Mycosphaerella musicola", font=fontR(17), fill=LGRAY, anchor="mm")
d.line([(200,308),(1000,308)], fill=ORANGE, width=1)

d.text((600,342), "Detection Confidence", font=fontR(15), fill=LGRAY, anchor="mm")
d.rounded_rectangle([(200,365),(1000,405)], radius=8, fill=(30,60,32))
d.rounded_rectangle([(200,365),(200+int(800*0.72),405)], radius=8, fill=ORANGE)
d.text((1010,385), "72%", font=font(18), fill=ACCENT, anchor="mm")

dinfo = [
    ("Disease Type","Fungal"),("Pathogen","M. musicola"),("Severity","Moderate"),
    ("Spread Risk","High\n(Humidity 58%)"),("Affected Part","Leaves"),("Confidence","Above Threshold"),
]
for i,(k,v) in enumerate(dinfo):
    col = i%3; row = i//3
    ix  = 200+col*280; iy = 445+row*110
    d.rounded_rectangle([(ix,iy),(ix+260,iy+90)], radius=10, fill=(30,60,32), outline=ORANGE, width=1)
    d.text((ix+130,iy+20), k, font=fontR(12), fill=LGRAY, anchor="mm")
    for j,ln in enumerate(v.split("\n")):
        d.text((ix+130,iy+50+j*18), ln, font=font(14), fill=WHITE, anchor="mm")

d.text((600,695), "⚠  Disease detected above confidence threshold. Full report generated.", font=fontR(15), fill=ORANGE, anchor="mm")
d.text((600,780), "Figure 4.5 – Disease Detection Card with Confidence Bar (Yellow Sigatoka, 72%)", font=fontR(13), fill=LGRAY, anchor="mm")
save(img, 45, "Disease_Detection_Card")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4.6 – AI Diagnostic Report
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 900)
d.rectangle([(0,0),(1200,900)], fill=(20,40,22))

d.rounded_rectangle([(60,80),(1140,840)], radius=16, fill=(24,48,26), outline=GREEN, width=2)
d.rounded_rectangle([(60,80),(1140,155)], radius=16, fill=DGREEN)
d.text((200,117), "🤖  AI Diagnostic Report", font=font(22), fill=WHITE, anchor="mm")
d.text((700,117), "Generated by Google Gemini 1.5 Flash", font=fontR(14), fill=LGREEN, anchor="mm")

sections = [
    ("1. Plant Identified", LGREEN,
     "Banana (Musa paradisiaca). The uploaded leaf belongs to the common\nbanana plant, widely grown across tropical and subtropical regions."),
    ("2. Likely Disease / Health Status", ORANGE,
     "Yellow Sigatoka (Mycosphaerella musicola). The leaf shows characteristic\nyellow streaks and spots consistent with Sigatoka fungal infection."),
    ("3. Confidence & Uncertainty Note", LBLUE,
     "Identification confidence: 87%  •  Disease confidence: 72%.\nResults are reliable; environmental context confirms elevated spread risk."),
    ("4. Likely Cause", ACCENT,
     "Fungal spores spread by wind and rain during humid conditions.\nCurrent humidity of 58% and recent rainfall have likely accelerated spread."),
    ("5. Treatment (Plain English)", LGREEN,
     "Apply copper-based fungicide or mancozeb spray to affected leaves.\nRemove and destroy heavily infected leaves. Repeat every 10–14 days."),
    ("6. Weather-Aware Prevention Tips", (150,220,255),
     "Given current temperature (34°C) and humidity (58%), monitor plants\ndaily. Improve air circulation. Avoid overhead irrigation in mornings."),
]
for i,(title,col,text) in enumerate(sections):
    row,c = divmod(i,2)
    sx = 80+c*540; sy = 170+row*210
    d.rounded_rectangle([(sx,sy),(sx+510,sy+195)], radius=10, fill=(28,56,30), outline=col, width=2)
    d.rounded_rectangle([(sx,sy),(sx+510,sy+38)], radius=10, fill=col)
    d.text((sx+12,sy+19), title, font=font(14), fill=WHITE if col!=ACCENT else DGREEN)
    for j,ln in enumerate(text.split("\n")):
        d.text((sx+12, sy+52+j*24), ln, font=fontR(13), fill=LGRAY)

d.text((600,860), "Figure 4.6 – Full AI Diagnostic Report Generated by Gemini 1.5 Flash", font=fontR(14), fill=LGRAY, anchor="mm")
save(img, 46, "AI_Diagnostic_Report")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4.7 – Sidebar Navigation
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
d.rectangle([(0,0),(1200,820)], fill=(20,40,22))

# Sidebar panel
d.rounded_rectangle([(30,80),(340,760)], radius=14, fill=(26,52,28), outline=GREEN, width=2)
d.text((185,110), "🌿 GreenLeaf", font=font(18), fill=LGREEN, anchor="mm")
d.line([(50,130),(320,130)], fill=GREEN, width=1)

items = [
    ("📊","Dashboard","Active"),
    ("🔍","Analyze Leaf",""),
    ("📋","My Reports",""),
    ("🌤","Weather",""),
    ("⚙️","Settings",""),
    ("❓","Help & Tips",""),
    ("👨‍💻","Developer Info",""),
]
for i,(icon,lbl,status) in enumerate(items):
    sy = 150+i*72
    bg = (34,75,38) if status=="Active" else (28,55,30)
    ol = LGREEN if status=="Active" else (40,80,45)
    d.rounded_rectangle([(50,sy),(320,sy+55)], radius=8, fill=bg, outline=ol, width=1 if not status else 2)
    d.text((78,sy+27), icon, font=font(18), fill=LGREEN, anchor="mm")
    d.text((200,sy+27), lbl, font=fontR(15), fill=WHITE if status else LGRAY, anchor="mm")
    if status: d.rounded_rectangle([(298,sy+16),(318,sy+39)], radius=4, fill=LGREEN)

d.text((185,710), "v1.0.0  •  UEM Jaipur 2025", font=fontR(11), fill=(80,120,80), anchor="mm")

# Main content area
d.rounded_rectangle([(360,80),(1160,760)], radius=14, fill=(24,48,26), outline=GREEN, width=1)
d.text((760,120), "Developer Information", font=font(20), fill=LGREEN, anchor="mm")
d.line([(400,145),(1120,145)], fill=GREEN, width=1)
dev_info = [
    ("Project","GreenLeaf – AI Plant Disease Detection"),
    ("Developer","YOUR NAME"),
    ("Institution","University of Engineering & Management, Jaipur"),
    ("Guide","PROF. GUIDE NAME"),
    ("APIs Used","PlantNet v2  •  Google Gemini 1.5 Flash  •  OpenWeatherMap"),
    ("Framework","Streamlit (Python)"),
    ("Version","1.0.0  •  Final Year Project 2025"),
]
for i,(k,v) in enumerate(dev_info):
    sy = 175+i*70
    d.text((400,sy), f"{k}:", font=font(14), fill=LGREEN)
    d.text((400,sy+24), v, font=fontR(14), fill=WHITE)
    d.line([(400,sy+52),(1120,sy+52)], fill=(30,65,35), width=1)

d.text((600,790), "Figure 4.7 – Sidebar Navigation Panel Showing Developer Information and Usage Tips", font=fontR(13), fill=LGRAY, anchor="mm")
save(img, 47, "Sidebar_Navigation")

# ═══════════════════════════════════════════════════════════════════════════
# Figure 4.8 – Mobile Responsive Layout
# ═══════════════════════════════════════════════════════════════════════════
img, d = new_canvas(1200, 820)
d.rectangle([(0,0),(1200,820)], fill=(230,235,230))

# Phone frame
px, py, pw, ph = 400, 50, 400, 720
d.rounded_rectangle([(px,py),(px+pw,py+ph)], radius=40, fill=(15,15,20), outline=(80,80,80), width=6)
d.rounded_rectangle([(px+8,py+8),(px+pw-8,py+ph-8)], radius=34, fill=(20,40,22))
# notch
d.rounded_rectangle([(px+150,py+10),(px+250,py+34)], radius=10, fill=(10,10,15))

# Mobile screen content
d.text((px+200,py+55), "🌿 GreenLeaf", font=font(18), fill=LGREEN, anchor="mm")
d.rounded_rectangle([(px+20,py+75),(px+380,py+200)], radius=10, fill=(26,52,28), outline=GREEN, width=1)
d.text((px+200,py+115), "📷", font=font(28), fill=LGREEN, anchor="mm")
d.text((px+200,py+155), "Tap to upload leaf", font=fontR(13), fill=LGRAY, anchor="mm")
d.text((px+200,py+175), "or take a photo", font=fontR(12), fill=(100,140,100), anchor="mm")

d.rounded_rectangle([(px+20,py+215),(px+380,py+260)], radius=8, fill=(26,52,28), outline=GREEN, width=1)
d.text((px+40,py+237), "📍 Enter city name...", font=fontR(14), fill=(100,140,100))

# Weather badges (small)
d.rounded_rectangle([(px+20,py+275),(px+175,py+315)], radius=8, fill=(30,65,35), outline=LGREEN, width=1)
d.text((px+97,py+295), "🌡️ 34°C", font=fontR(13), fill=ACCENT, anchor="mm")
d.rounded_rectangle([(px+195,py+275),(px+380,py+315)], radius=8, fill=(30,65,35), outline=LGREEN, width=1)
d.text((px+287,py+295), "💧 58% RH", font=fontR(13), fill=LBLUE, anchor="mm")

d.rounded_rectangle([(px+20,py+330),(px+380,py+385)], radius=10, fill=GREEN)
d.text((px+200,py+357), "🔍  Analyze", font=font(18), fill=WHITE, anchor="mm")

# Results (compact)
d.text((px+200,py+410), "Plant: Banana (87%)", font=fontR(13), fill=LGREEN, anchor="mm")
d.text((px+200,py+435), "Disease: Yellow Sigatoka (72%)", font=fontR(13), fill=ORANGE, anchor="mm")
d.rounded_rectangle([(px+20,py+455),(px+380,py+560)], radius=8, fill=(26,52,28), outline=GREEN, width=1)
d.text((px+200,py+475), "AI Report Summary", font=font(13), fill=LGREEN, anchor="mm")
report_lines = [
    "Apply copper fungicide to",
    "affected leaves. Remove",
    "heavily infected parts.",
    "Improve air circulation.",
]
for i,ln in enumerate(report_lines):
    d.text((px+30,py+500+i*16), ln, font=fontR(11), fill=LGRAY)

# Home bar
d.rounded_rectangle([(px+160,py+685),(px+240,py+698)], radius=5, fill=(80,80,80))

# Labels outside phone
d.text((290,390), "Responsive\nLayout", font=font(16), fill=DGREEN)
d.text((290,440), "380px\nviewport", font=fontR(14), fill=DGRAY)
arrow(d, 398,400, 370,400, color=GREEN, w=2)

d.text((830,250), "✓  Touch-friendly", font=fontR(15), fill=GREEN)
d.text((830,285), "✓  Single-column layout", font=fontR(15), fill=GREEN)
d.text((830,320), "✓  No horizontal scroll", font=fontR(15), fill=GREEN)
d.text((830,355), "✓  Readable font sizes", font=fontR(15), fill=GREEN)
d.text((830,390), "✓  Large tap targets", font=fontR(15), fill=GREEN)
d.text((830,425), "✓  Works on 4G / WiFi", font=fontR(15), fill=GREEN)

d.text((600,790), "Figure 4.8 – Responsive Layout on Mobile Browser (Galaxy S21 Viewport)", font=fontR(14), fill=DGRAY, anchor="mm")
save(img, 48, "Mobile_Responsive")

print("\n✅ All 11 figures generated!")
