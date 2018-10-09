//
//  ViewController.m
//  MINST
//
//  Created by 夏潘安 on 9/15/30 H.
//  Copyright © 30 Heisei suponline.supcon.com. All rights reserved.
//

#import "ViewController.h"
#import "MNIST.h"
#import "PopSignatureView.h"

#define screenWidth     [UIScreen mainScreen].bounds.size.width
#define screenHeight    [UIScreen mainScreen].bounds.size.height

@interface ViewController ()<PopSignatureViewDelegate>

@property(nonatomic,strong)UIImageView *showImageView;

@property(nonatomic,strong)UILabel *outputLabel;

@end

@implementation ViewController

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
     UIImage *image = [UIImage imageNamed:@"three"];
    
    UIButton *signIn = [UIButton buttonWithType:UIButtonTypeSystem];
    signIn.frame = CGRectMake(20, 20, 100, 50);
    [signIn setTitle:@"写个数字" forState:UIControlStateNormal];
    [signIn addTarget:self action:@selector(signInClick) forControlEvents:UIControlEventTouchUpInside];
    [self.view addSubview:signIn];
    
    self.showImageView = [[UIImageView alloc]initWithFrame:CGRectMake(0, 80, screenWidth, screenWidth)];
    [self.showImageView setImage:image];
    [self.view addSubview:self.showImageView];
    
    [self mnistCheck:image];
    
}

-(void)signInClick{
    PopSignatureView *socialSingnatureView = [[PopSignatureView alloc] initWithFrame:CGRectMake(0, 0, screenWidth , screenHeight)];
    socialSingnatureView.delegate = self;
    [socialSingnatureView show];
}

- (void)onSubmitBtn:(UIImage *)signatureImg {
    NSLog(@"高：%f,宽：%f",signatureImg.size.height,signatureImg.size.width);
    
    signatureImg = [self inverColorImage:signatureImg];
    
    self.showImageView.image = signatureImg;
    
    if(signatureImg != nil){
        
        UIImage *xiaotu = [self OriginImage:signatureImg scaleToSize:CGSizeMake(28, 28)];
        
        [self mnistCheck:xiaotu];
    }
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (UILabel *)outputLabel{
    if(!_outputLabel){
        _outputLabel = [[UILabel alloc]initWithFrame:CGRectMake( 0, 100 + screenWidth, screenWidth, 40)];
        _outputLabel.textAlignment = NSTextAlignmentCenter;
        _outputLabel.textColor = [UIColor colorWithRed:0 green:0 blue:0 alpha:1];
        [_outputLabel setFont:[UIFont systemFontOfSize:23]];
        [self.view addSubview:_outputLabel];
    }
    return _outputLabel;
}

//进入model预测
- (void)mnistCheck:(UIImage *)image{
    MNIST *mMnistModel = [[MNIST alloc]init];
//    MNISTInput *input = [[MNISTInput alloc]initWithImage:[self pixelBufferFromCGImage:image.CGImage]];
    CVPixelBufferRef buffer = [self pixelBufferFromCGImage:image.CGImage];
    
    MNISTOutput * output = [MNISTOutput new];
    output = [mMnistModel predictionFromImage:buffer error:nil];
    
    NSLog(@"我猜是%lld",output.classLabel);
    NSLog(@"%@",output.prediction );
    
    self.outputLabel.text = [NSString stringWithFormat:@"我猜是:%lld",output.classLabel];
    
//    mMnistModel clear
}

//UIImage缩小
-(UIImage*)  OriginImage:(UIImage *)image   scaleToSize:(CGSize)size
{
    // 创建一个bitmap的context
    // 并把它设置成为当前正在使用的context
    UIGraphicsBeginImageContext(size);
    
    // 绘制改变大小的图片
    [image drawInRect:CGRectMake(0, 0, size.width, size.height)];
    
    // 从当前context中创建一个改变大小后的图片
    UIImage* scaledImage = UIGraphicsGetImageFromCurrentImageContext();
    
    // 使当前的context出堆栈
    UIGraphicsEndImageContext();
    
    // 返回新的改变大小后的图片
    return scaledImage;
}

// ColorInver
- (UIImage *)inverColorImage:(UIImage *)image {
    CGImageRef cgimage = image.CGImage;
    size_t width = CGImageGetWidth(cgimage);
    size_t height = CGImageGetHeight(cgimage);
    
    // 取图片首地址
    unsigned char *data = calloc(width * height * 4, sizeof(unsigned char));
    size_t bitsPerComponent = 8; // r g b a 每个component bits数目
    size_t bytesPerRow = width * 4; // 一张图片每行字节数目 (每个像素点包含r g b a 四个字节)
    // 创建rgb颜色空间
    CGColorSpaceRef space = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(data, width, height, bitsPerComponent, bytesPerRow, space, kCGImageAlphaPremultipliedLast | kCGBitmapByteOrder32Big);
    CGContextDrawImage(context, CGRectMake(0, 0, width, height), cgimage);
    
    for (size_t i = 0; i < height; i++) {
        for (size_t j = 0; j < width; j++) {
            size_t pixelIndex = i * width * 4 + j * 4;
            unsigned char red = data[pixelIndex];
            unsigned char green = data[pixelIndex + 1];
            unsigned char blue = data[pixelIndex + 2];
            // 修改颜色
            data[pixelIndex] = 255 - red;
            data[pixelIndex + 1] = 255 - green;
            data[pixelIndex + 2] = 255 - blue;
            
        }
    }
    cgimage = CGBitmapContextCreateImage(context);
    return [UIImage imageWithCGImage:cgimage];
}

//UIImage变OneComponent8
- (CVPixelBufferRef) pixelBufferFromCGImage: (CGImageRef) image {
    NSDictionary *options = @{
                              (NSString*)kCVPixelBufferCGImageCompatibilityKey : @YES,
                              (NSString*)kCVPixelBufferCGBitmapContextCompatibilityKey : @YES,
                              };
    
    CVPixelBufferRef pxbuffer = NULL;
    CVReturn status = CVPixelBufferCreate(kCFAllocatorDefault, CGImageGetWidth(image),
                                          CGImageGetHeight(image), kCVPixelFormatType_OneComponent8, (__bridge CFDictionaryRef) options,
                                          &pxbuffer);
    if (status!=kCVReturnSuccess) {
        NSLog(@"Operation failed");
    }
    NSParameterAssert(status == kCVReturnSuccess && pxbuffer != NULL);
    
    CVPixelBufferLockBaseAddress(pxbuffer, 0);
    void *pxdata = CVPixelBufferGetBaseAddress(pxbuffer);
    
    CGColorSpaceRef rgbColorSpace = CGColorSpaceCreateDeviceGray(); //灰度
    CGContextRef context = CGBitmapContextCreate(pxdata, CGImageGetWidth(image),
                                                 CGImageGetHeight(image), 8, CVPixelBufferGetBytesPerRow(pxbuffer), rgbColorSpace,
                                                 kCGImageAlphaNone);
    NSParameterAssert(context);
    CGContextConcatCTM(context, CGAffineTransformMakeRotation(0));
    CGContextDrawImage(context, CGRectMake(0, 0, CGImageGetWidth(image),
                                           CGImageGetHeight(image)), image);
    CGColorSpaceRelease(rgbColorSpace);
    CGContextRelease(context);
    CVPixelBufferUnlockBaseAddress(pxbuffer, 0);
    return pxbuffer;
}

@end
