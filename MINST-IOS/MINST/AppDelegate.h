//
//  AppDelegate.h
//  MINST
//
//  Created by 夏潘安 on 9/15/30 H.
//  Copyright © 30 Heisei suponline.supcon.com. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <CoreData/CoreData.h>

@interface AppDelegate : UIResponder <UIApplicationDelegate>

@property (strong, nonatomic) UIWindow *window;

@property (readonly, strong) NSPersistentContainer *persistentContainer;

- (void)saveContext;


@end

