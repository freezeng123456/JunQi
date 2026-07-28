/*
 * board.c
 *
 *  Created on: Jul 28, 2018
 *      Author: Administrator
 */
#include <gtk/gtk.h>
#include <stdlib.h>
#include <string.h>
#include "junqi.h"
#include <pthread.h>
#include <unistd.h>
#include "rule.h"
#include "comm.h"

static GdkPixbuf *background;
static Junqi *gJunqi;

#ifndef _WIN32
#include <errno.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>

#define SND_FILENAME 0x00020000L
#define SND_ASYNC    0x00000001L
#define SND_LOOP     0x00000008L
#define SND_NODEFAULT 0x00000002L

static pthread_mutex_t gSoundMutex = PTHREAD_MUTEX_INITIALIZER;
static pid_t gSoundPid = -1;

static void TerminateSoundProcess(pid_t pid)
{
	int i;
	pid_t result;

	if(pid <= 0)
		return;
	(void)kill(pid, SIGTERM);
	for(i = 0; i < 20; i++)
	{
		result = waitpid(pid, NULL, WNOHANG);
		if(result == pid || (result < 0 && errno == ECHILD))
			return;
		usleep(5000);
	}
	/* afplay should exit on SIGTERM; SIGKILL is a bounded fallback so the
	 * mute action never blocks the GTK UI indefinitely. */
	(void)kill(pid, SIGKILL);
	(void)waitpid(pid, NULL, 0);
}

static void StopActiveSound(void)
{
	pid_t pid;

	pthread_mutex_lock(&gSoundMutex);
	pid = gSoundPid;
	gSoundPid = -1;
	if(pid > 0)
		TerminateSoundProcess(pid);
	pthread_mutex_unlock(&gSoundMutex);
}

void PlaySound(const char* pszSound, void* hmod, int fdwSound) {
    pid_t pid;

    (void)hmod;
    (void)fdwSound;
    if(pszSound == NULL || (gJunqi != NULL && gJunqi->bMute))
        return;

    /* The old implementation used system("afplay ... &"), which detached
     * every sound process and made it impossible to stop queued audio. */
    pthread_mutex_lock(&gSoundMutex);
    if(gSoundPid > 0)
    {
        TerminateSoundProcess(gSoundPid);
        gSoundPid = -1;
    }
    pid = fork();
    if(pid == 0)
    {
        execlp("afplay", "afplay", pszSound, (char *)NULL);
        _exit(127);
    }
    if(pid > 0)
        gSoundPid = pid;
    pthread_mutex_unlock(&gSoundMutex);
}

#define Sleep(ms) usleep((ms) * 1000)
#else
static void StopActiveSound(void) {}
#endif

typedef struct BoardItem
{
	GtkWidget *lineup_button[4];
	GtkWidget *surrender_button[4];
	GtkWidget *jump_button[4];
	GtkWidget *open_menu;
	GtkWidget *save_menu;

}BoardItem;

typedef struct DialogArg
{
	Junqi *pJunqi;
	char str[100];
	int num;
}DialogArg;

BoardItem gBoard;



void SetTimeStr(char *label_str, int time);

gboolean close_dialog(gpointer data)
{
	gtk_dialog_response(data, GTK_RESPONSE_OK);

	return 0;
}

gboolean dialog_event(gpointer data)
{
	GtkWidget *dialog;
	DialogArg *pArg = (DialogArg*)data;

	guint timer_id;
	gint  response_id;

	dialog = gtk_message_dialog_new (GTK_WINDOW (pArg->pJunqi->window),
									 GTK_DIALOG_DESTROY_WITH_PARENT,
									 GTK_MESSAGE_INFO,
									 GTK_BUTTONS_NONE,
									 "%s", pArg->str);
	if( pArg->num!=0 )
		gtk_message_dialog_format_secondary_text (GTK_MESSAGE_DIALOG (dialog),
												  "%d", pArg->num);
	//2秒后如果还没关闭对话框，则自动关闭对话框
	timer_id = g_timeout_add(1000, (GSourceFunc)close_dialog, dialog);
	response_id = gtk_dialog_run (GTK_DIALOG (dialog));
	if(response_id != GTK_RESPONSE_OK)
	{
		g_source_remove(timer_id);
	}
	gtk_widget_destroy (dialog);
	free(pArg);
	//定时器返回0只执行一次
	return FALSE;
}

//str是字符内容，num为0则不显示数字
void ShowDialogMessage(Junqi *pJunqi, char *str, int num)
{
	DialogArg *pArg = (DialogArg*)malloc(sizeof(DialogArg));
	if(pArg == NULL || pJunqi == NULL || str == NULL)
	{
		free(pArg);
		return;
	}
	memset(pArg, 0, sizeof(DialogArg));

	pArg->pJunqi = pJunqi;
	snprintf(pArg->str, sizeof(pArg->str), "%s", str);
	pArg->num = num;
    //把对话框的显示放在定时器里处理，否则这里会被阻塞，那么时间就会停顿
	g_timeout_add(100, (GSourceFunc)dialog_event, pArg);
}

void ResetBoardButton(Junqi *pJunqi)
{
	gtk_widget_hide(pJunqi->start_button);
	gtk_widget_show(pJunqi->begin_button);
	for(int i=0; i<4; i++)
	{
		gtk_widget_show(gBoard.lineup_button[i]);
		gtk_widget_hide(gBoard.surrender_button[i]);
		gtk_widget_hide(gBoard.jump_button[i]);
	}
	gtk_widget_hide(pJunqi->step_fixed);
}

void ShowReplaySlider(Junqi *pJunqi)
{
	gtk_widget_hide(pJunqi->begin_button);
	for(int i=0; i<4; i++)
	{
		gtk_widget_hide(gBoard.lineup_button[i]);
	}
	gtk_widget_show(pJunqi->step_fixed);
}


void CreatSaveDialog(Junqi *pJunqi, GCallback c_handler)
{
    GtkFileChooserNative *native;

	native = gtk_file_chooser_native_new ("Save File",
										NULL,
										GTK_FILE_CHOOSER_ACTION_SAVE,
										"_Save",
										"_Cancel");
	gtk_native_dialog_show (GTK_NATIVE_DIALOG (native));
	pJunqi->native = native;
	g_signal_connect (native,
					"response",
					c_handler,
					pJunqi);

}

void CreaOpenDialog(Junqi *pJunqi)
{
    GtkFileChooserNative *native;

	native = gtk_file_chooser_native_new ("Open File",
										NULL,
										GTK_FILE_CHOOSER_ACTION_OPEN,
										"_Open",
										"_Cancel");
	gtk_native_dialog_show (GTK_NATIVE_DIALOG (native));
	pJunqi->native = native;
	g_signal_connect (native,
					"response",
					G_CALLBACK (OpenReplay),
					pJunqi);

}

static void event_handle(GtkWidget *item,gpointer data)
{
	char *event = (char*)data;
	Junqi *pJunqi = gJunqi;
	if( event==NULL )
	{
		return;
	}
	if( strcmp(event,"new" )==0 )
	{
		pJunqi->bStart = 0;
		pJunqi->bReplay = 0;
		pJunqi->bStop = 0;
		pJunqi->bAnalyse = 0;
		pJunqi->bResetFlag = 1;
		pJunqi->eTurn = pJunqi->eFirstTurn;
		ReSetChessBoard(pJunqi);
		DestroyChessFlag(pJunqi);
		ResetBoardButton(pJunqi);
		gtk_widget_set_sensitive(gBoard.open_menu, TRUE);
		gtk_widget_set_sensitive(gBoard.save_menu, FALSE);

		SendHeader(pJunqi, 0, COMM_READY);
	    SendHeader(pJunqi, 1, COMM_READY);

	}
	else if( strcmp(event,"save" )==0 )
	{
		CreatSaveDialog(pJunqi, G_CALLBACK (SaveReplay));
	}
	else if( strcmp(event,"stop" )==0 )
	{
		pJunqi->bStop = 1;

		SendHeader(pJunqi, 0, COMM_STOP);
#ifndef NOT_DEBUG2
		SendHeader(pJunqi, 1, COMM_STOP);
#endif
	}
	else if( strcmp(event,"continue" )==0 )
	{
		pJunqi->bStop = 0;
		if( pJunqi->bStart )
		{
			SendHeader(pJunqi, 0, COMM_GO);
			SendHeader(pJunqi, 1, COMM_GO);
		}
	}
	else if( strcmp(event,"open" )==0 )
	{
		CreaOpenDialog(pJunqi);
	}
	else if(  strcmp(event,"analyse" )==0 )
	{
		if( pJunqi->bReplay )
		{
			pJunqi->bReplay = 0;
			pJunqi->bAnalyse = 1;
			SendHeader(pJunqi, 0, COMM_READY);
		    SendHeader(pJunqi, 1, COMM_READY);
		}
	}
}

GtkWidget *SetMenuItem(GtkWidget *menu, char *zLabel, void *call_back, char *arg)
{
	GtkWidget *menuitem;
	menuitem=gtk_menu_item_new_with_label(zLabel);
	gtk_menu_shell_append (GTK_MENU_SHELL (menu), menuitem);
	g_signal_connect(GTK_MENU_ITEM(menuitem),"activate",G_CALLBACK(call_back),arg);
	return menuitem;
}

int GetColorDir(Junqi *pJunqi, char *zDir)
{
	int iDir = 0;
    if( strcmp(zDir,"orange" )==0 )
    {
    	iDir = (4-pJunqi->eColor)%4;
    }
    else if( strcmp(zDir,"purple" )==0 )
    {
    	iDir = (4-pJunqi->eColor+1)%4;
    }
    else if( strcmp(zDir,"green" )==0 )
    {
    	iDir = (4-pJunqi->eColor+2)%4;
    }
    else if( strcmp(zDir,"blue" )==0 )
    {
    	iDir = (4-pJunqi->eColor+3)%4;
    }
    return iDir;
}

static void save_lineup_handle(GtkWidget *item,gpointer data)
{
	enum ChessDir iDir;
	Junqi *pJunqi = gJunqi;
	char *zDir = (char*)data;

	if(zDir==NULL) return;

	iDir = GetColorDir(pJunqi,zDir);

    pJunqi->eLineupDir = iDir;

    CreatSaveDialog(pJunqi, G_CALLBACK (SaveLineup));
}

static void set_first_turn(GtkWidget *item,gpointer data)
{
	enum ChessDir iDir;
	Junqi *pJunqi = gJunqi;
	char *zDir = (char*)data;

	if(zDir==NULL) return;

	iDir = GetColorDir(pJunqi,zDir);
	pJunqi->eTurn = iDir;
	pJunqi->eFirstTurn = iDir;
}

void CreatSaveLineupMenu(GtkWidget *menu)
{
	GtkWidget *menuitem,*save_menu;
	menuitem = gtk_menu_item_new_with_label("导出布阵");
	gtk_menu_shell_append (GTK_MENU_SHELL (menu), menuitem);
	save_menu = gtk_menu_new();
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(menuitem),save_menu);
	SetMenuItem(save_menu, "橙色", save_lineup_handle, "orange");
	SetMenuItem(save_menu, "绿色", save_lineup_handle, "green");
	SetMenuItem(save_menu, "蓝色", save_lineup_handle, "blue");
	SetMenuItem(save_menu, "紫色", save_lineup_handle, "purple");
}

void CreatFirstTurnMenu(GtkWidget *menu)
{
	GtkWidget *menuitem,*sub_menu;
	menuitem = gtk_menu_item_new_with_label("先手");
	gtk_menu_shell_append (GTK_MENU_SHELL (menu), menuitem);
	sub_menu = gtk_menu_new();
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(menuitem),sub_menu);
	SetMenuItem(sub_menu, "橙色", set_first_turn, "orange");
	SetMenuItem(sub_menu, "绿色", set_first_turn, "green");
	SetMenuItem(sub_menu, "蓝色", set_first_turn, "blue");
	SetMenuItem(sub_menu, "紫色", set_first_turn, "purple");
}

void SaveConfig(Junqi *pJunqi)
{
    FILE *fp = fopen("config.ini", "w");
    if(fp)
    {
        fprintf(fp, "show_mode=%d\n", pJunqi->eShowMode);
        fprintf(fp, "mute=%d\n", pJunqi->bMute ? 1 : 0);
        fclose(fp);
    }
}

void LoadConfig(Junqi *pJunqi)
{
    FILE *fp = fopen("config.ini", "r");
    pJunqi->eShowMode = SHOW_MODE_DARK; // 默认暗棋
    pJunqi->bMute = 0;
    if(fp)
    {
        char key[32];
        int value;
        while(fscanf(fp, " %31[^=]=%d", key, &value) == 2)
        {
            if(strcmp(key, "show_mode") == 0 &&
                    value >= SHOW_MODE_BRIGHT && value <= SHOW_MODE_HALF_DARK)
            {
                pJunqi->eShowMode = (enum ShowMode)value;
            }
            else if(strcmp(key, "mute") == 0)
            {
                pJunqi->bMute = value ? 1 : 0;
            }
        }
        fclose(fp);
    }
}

static void set_show_mode(GtkWidget *item, gpointer data)
{
    Junqi *pJunqi = gJunqi;
    char *mode_str = (char*)data;
    if(strcmp(mode_str, "bright") == 0)
    {
        pJunqi->eShowMode = SHOW_MODE_BRIGHT;
    }
    else if(strcmp(mode_str, "half_dark") == 0)
    {
        pJunqi->eShowMode = SHOW_MODE_HALF_DARK;
    }
    else
    {
        pJunqi->eShowMode = SHOW_MODE_DARK;
    }
    SaveConfig(pJunqi);
    // 刷新棋盘显示
    RefreshChessBoardDisplay(pJunqi);
}

void CreatShowModeMenu(GtkWidget *menu)
{
	GtkWidget *menuitem,*sub_menu;
	menuitem = gtk_menu_item_new_with_label("显示模式");
	gtk_menu_shell_append (GTK_MENU_SHELL (menu), menuitem);
	sub_menu = gtk_menu_new();
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(menuitem),sub_menu);
	SetMenuItem(sub_menu, "明棋(全可见)", set_show_mode, "bright");
	SetMenuItem(sub_menu, "暗棋(隐藏非己方)", set_show_mode, "dark");
	SetMenuItem(sub_menu, "半暗棋(隐藏左右)", set_show_mode, "half_dark");
}

static void set_mute(GtkCheckMenuItem *item, gpointer data)
{
	Junqi *pJunqi = (Junqi *)data;

	pJunqi->bMute = gtk_check_menu_item_get_active(item) ? 1 : 0;
	if(pJunqi->bMute)
	{
		/* Discard events that were queued before the user muted the client. */
		pJunqi->sound_type = 0;
		pJunqi->sound_replay = 0;
		pJunqi->szPathForSound = 0;
		pJunqi->szPath = 0;
		StopActiveSound();
	}
	SaveConfig(pJunqi);
}

static void CreatMuteMenu(GtkWidget *menu, Junqi *pJunqi)
{
	GtkWidget *menuitem = gtk_check_menu_item_new_with_label("静音");
	gtk_check_menu_item_set_active(GTK_CHECK_MENU_ITEM(menuitem), pJunqi->bMute);
	gtk_menu_shell_append(GTK_MENU_SHELL(menu), menuitem);
	g_signal_connect(menuitem, "toggled", G_CALLBACK(set_mute), pJunqi);
}

/*
 * 设置菜单栏
 */
void set_menu(GtkWidget *vbox)
{
	GtkWidget *menubar,*menu,*menuitem;

    menubar=gtk_menu_bar_new();
    gtk_widget_set_hexpand (menubar, TRUE);
    gtk_box_pack_start (GTK_BOX (vbox), menubar, FALSE, TRUE, 0);
	menuitem=gtk_menu_item_new_with_label("文件");
	gtk_menu_shell_append (GTK_MENU_SHELL (menubar), menuitem);

	menu=gtk_menu_new();
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(menuitem),menu);
	SetMenuItem(menu, "新建", event_handle, "new");
	gBoard.open_menu = SetMenuItem(menu, "打开复盘", event_handle, "open");
	gBoard.save_menu = SetMenuItem(menu, "保存复盘", event_handle, "save");
	gtk_widget_set_sensitive(gBoard.save_menu, FALSE);
	CreatSaveLineupMenu(menu);

	menuitem=gtk_menu_item_new_with_label("设定");
	gtk_menu_shell_append (GTK_MENU_SHELL (menubar), menuitem);
	menu=gtk_menu_new();
	gtk_menu_item_set_submenu(GTK_MENU_ITEM(menuitem),menu);


	SetMenuItem(menu, "暂停", event_handle, "stop");
	SetMenuItem(menu, "继续", event_handle, "continue");
	CreatFirstTurnMenu(menu);
	CreatShowModeMenu(menu);
	CreatMuteMenu(menu, gJunqi);
	SetMenuItem(menu, "分析", event_handle, "analyse");
}

/*
 * 把背景图片显示在画布上，如果要要把画布显示到界面，此回调函数是必须的
 */
static gint draw_cb (
	 GtkWidget *widget,
	 cairo_t   *cr,
	 gpointer   data)
{
  gdk_cairo_set_source_pixbuf (cr, background, 0, 0);
  cairo_paint (cr);

  return TRUE;
}

/*
 * 根据坐标和长度截取pixbuf，并返回图片控件
 */
GtkWidget *get_image_from_pixbuf(GdkPixbuf* pixbuf,
		gint src_x,
		gint src_y,
		gint width,
		gint height )
{
	GtkWidget *image;
	GdkPixbuf* pTemp;
	pTemp=gdk_pixbuf_new_subpixbuf(pixbuf, src_x,src_y,width,height);
	image = gtk_image_new_from_pixbuf(pTemp);
	g_object_unref (pTemp);
	return image;
}

GtkWidget *get_image_from_name(char *zName,
		gint src_x,
		gint src_y,
		gint width,
		gint height )
{
	GtkWidget *image;
	GdkPixbuf* pixbuf;
	pixbuf = gdk_pixbuf_new_from_file(zName, NULL);
	image = get_image_from_pixbuf(pixbuf, src_x,src_y,width,height);
	g_object_unref (pixbuf);
	return image;
}

enum ChessDir GetButtonDir(char* zDir)
{
	enum ChessDir iDir = HOME;
    if( strcmp(zDir,"home" )==0 )
    {
    	iDir = HOME;
    }
    else if( strcmp(zDir,"right" )==0 )
    {
    	iDir = RIGHT;
    }
    else if( strcmp(zDir,"opps" )==0 )
    {
    	iDir = OPPS;
    }
    else if( strcmp(zDir,"left" )==0 )
    {
    	iDir = LEFT;
    }

    return iDir;
}

static void
on_slider_value_change (GtkAdjustment *adjustment,
                      gpointer       data)
{
	Junqi *pJunqi = (Junqi*)data;
	int value;
	int max_step = 0;
	memcpy(&max_step, &pJunqi->aReplay[4], sizeof(max_step));

	value = gtk_adjustment_get_value (adjustment);
	assert( value>=0 && value<=max_step );
	if( value!=pJunqi->iRpStep )
	{
		pJunqi->bAnalyse = 0;
		pJunqi->iRpStep = value;
		ShowReplayStep(pJunqi, 0);
		pJunqi->sound_type = 0;
	}

}

static void step_cb(GtkWidget *button , gpointer data)
{
    char *zDir = (char*)data;
    Junqi *pJunqi = gJunqi;
	int max_step = 0;
	memcpy(&max_step, &pJunqi->aReplay[4], sizeof(max_step));
    if( strcmp(zDir,"left" )==0 )
    {
    	if( pJunqi->iRpStep>0 )
    	{
    		gtk_adjustment_set_value(pJunqi->slider_adj, pJunqi->iRpStep-1);
    	}

    }
    else if( strcmp(zDir,"right" )==0 )
    {
    	if( pJunqi->iRpStep<max_step )
    	{
    		pJunqi->iRpStep++;
    		gtk_adjustment_set_value(pJunqi->slider_adj, pJunqi->iRpStep);
    		ShowReplayStep(pJunqi, 1);
    	}
    }

}

static void button_cb(GtkWidget *button , gpointer data)
{
    char *zDir = (char*)data;
    Junqi *pJunqi = gJunqi;
    GtkFileChooserNative *native;

    assert(zDir!=NULL);
    pJunqi->selectDir = GetButtonDir(zDir);

	native = gtk_file_chooser_native_new ("Open File",
										NULL,
										GTK_FILE_CHOOSER_ACTION_OPEN,
										"_Open",
										"_Cancel");
	gtk_native_dialog_show (GTK_NATIVE_DIALOG (native));
	pJunqi->native = native;
	g_signal_connect (native,
					"response",
					G_CALLBACK (get_lineup_cb),
					pJunqi);
}

void HideJumpButton(int iDir)
{
	gtk_widget_hide(gBoard.surrender_button[iDir]);
	gtk_widget_hide(gBoard.jump_button[iDir]);
}

static void  surrender_cb(GtkWidget *button , gpointer data)
{
	GtkWidget *dialog;
	GtkWidget *content_area;
	GtkWidget *label;
	gint response;
	Junqi *pJunqi = gJunqi;
	char *zDir = (char*)data;
	int iDir;

	iDir = GetButtonDir(zDir);
	if( iDir!=(int)pJunqi->eTurn )
	{
		return;
	}
	dialog = gtk_dialog_new_with_buttons ("Interactive Dialog",
										GTK_WINDOW (pJunqi->window),
										GTK_DIALOG_MODAL| GTK_DIALOG_DESTROY_WITH_PARENT,
										("_OK"),
										GTK_RESPONSE_OK,
										"_Cancel",
										GTK_RESPONSE_CANCEL,
										NULL);
	content_area = gtk_dialog_get_content_area (GTK_DIALOG (dialog));
	label = gtk_label_new_with_mnemonic ("您确定投降吗");
	gtk_box_pack_start (GTK_BOX (content_area), label, FALSE, FALSE, 0);
	gtk_widget_show(label);
	response = gtk_dialog_run (GTK_DIALOG (dialog));
	if (response == GTK_RESPONSE_OK)
	{
		pJunqi->addr = pJunqi->addr_tmp[0];
		SendEvent(pJunqi, iDir, SURRENDER_EVENT);
#ifndef NOT_DEBUG2
		pJunqi->addr = pJunqi->addr_tmp[1];
		SendEvent(pJunqi, iDir, SURRENDER_EVENT);
#endif
		SendSoundEvent(pJunqi,DEAD);
		DestroyAllChess(pJunqi, iDir);
		ClearChessFlag(pJunqi,iDir);
		AddEventToReplay(pJunqi, SURRENDER_EVENT, iDir);
		ChessTurn(pJunqi);

		HideJumpButton(iDir);
	}

	gtk_widget_destroy (dialog);

}

static void  jump_cb(GtkWidget *button , gpointer data)
{
	Junqi *pJunqi = gJunqi;
	char *zDir = (char*)data;
	int iDir;
	iDir = GetButtonDir(zDir);
	if( iDir==(int)pJunqi->eTurn )
	{
		AddEventToReplay(pJunqi, JUMP_EVENT, iDir);
		IncJumpCnt(pJunqi, iDir);
		ChessTurn(pJunqi);

		pJunqi->addr = pJunqi->addr_tmp[0];
		SendEvent(pJunqi, iDir, JUMP_EVENT);
#ifndef NOT_DEBUG2
		pJunqi->addr = pJunqi->addr_tmp[1];
		SendEvent(pJunqi, iDir, JUMP_EVENT);
#endif
	}
}

/*
 * 设置调入布局按钮，每家都有一个
 */
void SetButton(GtkWidget *window, Junqi *pJunqi)
{
	GtkWidget **button = gBoard.lineup_button;
	GtkWidget **button1 = gBoard.surrender_button;
	GtkWidget **button2 = gBoard.jump_button;
	GError*  error =NULL;
	GdkPixbuf *pixbuf;
	GdkPixbuf *pixbuf1;
	GtkWidget *fixed = pJunqi->fixed;

	char *name =  "./res/load.bmp";
	char *name1 = "./res/jump.PNG";
	int i;
	for(i=0; i<4; i++)
	{
		button[i] = gtk_button_new();
		button1[i] = gtk_button_new();
		button2[i] = gtk_button_new();

		pixbuf = gdk_pixbuf_new_from_file(name, &error);
		pixbuf1 = gdk_pixbuf_new_from_file(name1, &error);
		GtkWidget *image =  get_image_from_pixbuf(pixbuf, 0,0,77,22);
		GtkWidget *image1 = get_image_from_pixbuf(pixbuf1, 7,92,88,28);
		GtkWidget *image2 = get_image_from_pixbuf(pixbuf1, 7,10,88,28);

		gtk_button_set_image(GTK_BUTTON(button[i]), image);
		gtk_button_set_relief(GTK_BUTTON(button[i]),GTK_RELIEF_NONE);

		gtk_button_set_image(GTK_BUTTON(button1[i]), image1);
		gtk_button_set_relief(GTK_BUTTON(button1[i]),GTK_RELIEF_NONE);

		gtk_button_set_image(GTK_BUTTON(button2[i]), image2);
		gtk_button_set_relief(GTK_BUTTON(button2[i]),GTK_RELIEF_NONE);
	    switch(i)
	    {
	    case 0:
	    	gtk_fixed_put(GTK_FIXED(fixed), button[i], 470,621);
	    	g_signal_connect (button[i], "clicked", G_CALLBACK (button_cb), "home");
	    	gtk_fixed_put(GTK_FIXED(fixed), button1[i], 470,621);
	    	g_signal_connect (button1[i], "clicked", G_CALLBACK (surrender_cb), "home");
	    	gtk_fixed_put(GTK_FIXED(fixed), button2[i], 470,591);
	    	g_signal_connect (button2[i], "clicked", G_CALLBACK (jump_cb), "home");
	    	break;
	    case 1:
	    	gtk_fixed_put(GTK_FIXED(fixed), button[i], 614,202);
	    	g_signal_connect (button[i], "clicked", G_CALLBACK (button_cb), "right");
	    	gtk_fixed_put(GTK_FIXED(fixed), button1[i], 614,202);
	    	g_signal_connect (button1[i], "clicked", G_CALLBACK (surrender_cb), "right");
	    	gtk_fixed_put(GTK_FIXED(fixed), button2[i], 614,172);
	    	g_signal_connect (button2[i], "clicked", G_CALLBACK (jump_cb), "right");
	    	break;
	    case 2:
	    	gtk_fixed_put(GTK_FIXED(fixed), button[i], 169,40);
	    	g_signal_connect (button[i], "clicked", G_CALLBACK (button_cb), "opps");
	    	gtk_fixed_put(GTK_FIXED(fixed), button1[i], 169,40);
	    	g_signal_connect (button1[i], "clicked", G_CALLBACK (surrender_cb), "opps");
	    	gtk_fixed_put(GTK_FIXED(fixed), button2[i], 169,10);
	    	g_signal_connect (button2[i], "clicked", G_CALLBACK (jump_cb), "opps");
	    	break;
	    case 3:
	    	gtk_fixed_put(GTK_FIXED(fixed), button[i], 60,452);
	    	g_signal_connect (button[i], "clicked", G_CALLBACK (button_cb), "left");
	    	gtk_fixed_put(GTK_FIXED(fixed), button1[i], 60,452);
	    	g_signal_connect (button1[i], "clicked", G_CALLBACK (surrender_cb), "left");
	    	gtk_fixed_put(GTK_FIXED(fixed), button2[i], 60,482);
	    	g_signal_connect (button2[i], "clicked", G_CALLBACK (jump_cb), "left");
	    	break;
	    default:
	    	break;
	    }
	    gtk_widget_show(button[i]);
	}
}


/*
 * 画一个矩形方框，用来选中棋子
 * isVertical：表示横竖
 * color：表示是白色还是红色
 */
GtkWidget *GetSelectImage(int isVertical, int color)
{
	cairo_t *cr;
	cairo_surface_t *surface = NULL;
	GdkPixbuf* pixbuf;
	GtkWidget *image;
	//新建一块画布，大小随意
	surface = cairo_image_surface_create ( CAIRO_FORMAT_ARGB32, 800, 900) ;
	cr = cairo_create (surface);

	cairo_set_line_width (cr, 2);
	if(color)
	{
		//红色方框
		cairo_set_source_rgb (cr, 255, 0, 0);
	}
	else
	{
		cairo_set_source_rgb (cr, 180, 238, 180);
	}
	cairo_rectangle (cr, 4.0, 4.0, 36.0, 27.0);
	cairo_stroke(cr);
	cairo_set_source_surface (cr, surface, 0, 0);
	cairo_paint (cr);

    pixbuf = gdk_pixbuf_get_from_surface(surface,0,0,50,50);
    if(isVertical)
    {
    	GdkPixbuf* src = pixbuf;
    	pixbuf = gdk_pixbuf_rotate_simple(src,GDK_PIXBUF_ROTATE_COUNTERCLOCKWISE);
    	g_object_unref (src);
    }
	image = gtk_image_new_from_pixbuf(pixbuf);

    cairo_surface_destroy (surface);
    cairo_destroy (cr);
    g_object_unref (pixbuf);

    return image;
}

static void send_go(GtkWidget *button, GdkEventButton *event, gpointer data)
{
	Junqi *pJunqi = (Junqi *)data;
	log_b("go %d",pJunqi->eTurn);
	if( pJunqi->bStart )
	{
		SendHeader(pJunqi, pJunqi->eTurn, COMM_GO);
		//SendHeader(pJunqi, 0, COMM_GO);
	}
}


static void begin_button(GtkWidget *button, GdkEventButton *event, gpointer data)
{
	Junqi *pJunqi = (Junqi *)data;
	//隐藏调入布局按钮
	for(int i=0; i<4; i++)
	{
		gtk_widget_hide(gBoard.lineup_button[i]);
		int show = 1;
		if((pJunqi->eShowMode == SHOW_MODE_DARK || pJunqi->eShowMode == SHOW_MODE_HALF_DARK) && i%2!=0)
		{
			show = 0;
		}
		if(show)
		{
			gtk_widget_show(gBoard.surrender_button[i]);
			gtk_widget_show(gBoard.jump_button[i]);
		}
	}


	pJunqi->bReplay = 0;
	pJunqi->bStart = 1;
	//pJunqi->bStop = 0;
	gtk_widget_hide(pJunqi->begin_button);
	gtk_widget_show(pJunqi->start_button);
	gtk_widget_set_sensitive(gBoard.open_menu, FALSE);
	gtk_widget_set_sensitive(gBoard.save_menu, TRUE);

	SendSoundEvent(pJunqi, BEGIN);

	pJunqi->iReOfst = 8;
	AddLineupToReplay(pJunqi);

	pJunqi->addr = pJunqi->addr_tmp[0];
	SendHeader(pJunqi, pJunqi->eFirstTurn, COMM_START);
#ifndef NOT_DEBUG2
	pJunqi->addr = pJunqi->addr_tmp[1];
	SendHeader(pJunqi, pJunqi->eFirstTurn, COMM_START);
#endif

}


void ShowTime( Junqi *pJunqi, int bClear )
{
	static int tick = 30;
	static int now_dir;
	static u8 bStart = 0;
	char label_str[100];

	if( bClear )
	{
		tick = 30;
		return;
	}
	if( now_dir!=(int)pJunqi->eTurn )
	{
		now_dir = pJunqi->eTurn;
		tick = 30;
	}
	SetTimeStr(label_str, tick);
	tick--;
	if( tick==0 )
	{
		SendEvent(pJunqi, pJunqi->eTurn, JUMP_EVENT);
		AddEventToReplay(pJunqi, JUMP_EVENT, pJunqi->eTurn);
		IncJumpCnt(pJunqi, pJunqi->eTurn);
		ChessTurn(pJunqi);
		now_dir = pJunqi->eTurn;
		tick = 30;
	}
    if( tick<10  )
    {
    	SendSoundEvent(pJunqi, TIMER);
    }

	gtk_label_set_markup(GTK_LABEL(pJunqi->pTimeLabel),label_str);
	if( !bStart )
	{
		bStart = 1;
		gtk_fixed_put(GTK_FIXED(pJunqi->fixed),pJunqi->pTimeLabel,221,628);
	}

	switch(pJunqi->eTurn)
	{
	case HOME:
		gtk_fixed_move(GTK_FIXED(pJunqi->fixed),pJunqi->pTimeLabel,221,628);
		break;
	case RIGHT:
		gtk_fixed_move(GTK_FIXED(pJunqi->fixed),pJunqi->pTimeLabel,654,446);
		break;
	case OPPS:
		gtk_fixed_move(GTK_FIXED(pJunqi->fixed),pJunqi->pTimeLabel,474,27);
		break;
	case LEFT:
		gtk_fixed_move(GTK_FIXED(pJunqi->fixed),pJunqi->pTimeLabel,47,206);
		break;
	default:
		break;
	}
	gtk_widget_show(pJunqi->pTimeLabel);

}

gboolean time_event(gpointer data)
{
	Junqi *pJunqi = (Junqi *)data;

    if( pJunqi->bStart && !pJunqi->bStop )
    {
    	//复盘时不要显示时间
    	if( !pJunqi->bReplay )
    		ShowTime(pJunqi, 0);
    }
    if( !pJunqi->bStart )
    {
    	ShowTime(pJunqi, 1);
    	gtk_widget_hide(pJunqi->pTimeLabel);

    }

    return 1;

}

void *sound_thread(void *arg)
{
	Junqi *pJunqi = (Junqi *)arg;
	enum CompareType current_type;

	while(1)
	{
		if(pJunqi->bMute)
		{
			pJunqi->sound_type = 0;
			pJunqi->sound_replay = 0;
			pJunqi->szPathForSound = 0;
			pJunqi->szPath = 0;
			StopActiveSound();
			Sleep(100);
			continue;
		}
		current_type = pJunqi->sound_type;
	    if( current_type>0 )
	    {
	        switch(current_type)
	        {
	        case MOVE:
		    	if( pJunqi->szPathForSound>2 )
		    	{
		    		PlaySound (MOVE_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_LOOP);
	    			Sleep(250);
		    	}
    			//异步播放后，非异步的移动声就播不出，此处做暂停用
    			PlaySound (MOVE_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
	        	break;
	        case BOMB:
	        	PlaySound (BOMB_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
	        	break;
	        case EAT:
	        	PlaySound (EAT_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
	        	break;
	        case KILLED:
	        	PlaySound (KILLED_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
	        	break;
	        case SELECT:
	        	PlaySound (SELECT_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
	        	break;
	        case SHOW_FLAG:
	        	PlaySound (SHOW_FLAG_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
	        	break;
	        case DEAD:
	        	PlaySound (DEAD_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
	        	break;
	        case BEGIN:
	        	PlaySound (BEGIN_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
	        	break;
	        case TIMER:
	        	PlaySound (TIMER_SOUND, NULL, SND_FILENAME | SND_ASYNC | SND_NODEFAULT);
	        	break;
	        default:
	        	break;
	        }
	        
	        // 只有当 sound_type 没有被外部修改时才清除
	        if( pJunqi->sound_type == current_type )
	        {
		        pJunqi->sound_type = 0;
		        pJunqi->szPathForSound = 0;
		        pJunqi->szPath = 0;
	        }
	    }

	    Sleep(100);
	    //usleep(100000);
	}
	return NULL;
}

static void stop_audio_on_destroy(GtkWidget *widget, gpointer data)
{
	Junqi *pJunqi = (Junqi *)data;
	(void)widget;
	if(pJunqi != NULL)
	{
		pJunqi->bMute = 1;
		pJunqi->sound_type = 0;
		pJunqi->sound_replay = 0;
		pJunqi->szPathForSound = 0;
		pJunqi->szPath = 0;
	}
	StopActiveSound();
}

void SendSoundEvent(Junqi *pJunqi, enum CompareType type)
{
	if(pJunqi->bMute)
	{
		pJunqi->sound_type = 0;
		pJunqi->sound_replay = 0;
		pJunqi->szPathForSound = 0;
		pJunqi->szPath = 0;
		return;
	}
	if( !pJunqi->bReplay )
	{
		pJunqi->sound_type = type;
	}
	else
	{
		pJunqi->sound_replay = type;
	}
	pJunqi->szPathForSound = pJunqi->szPath;
}


static GtkWidget *
add_section (GtkWidget   *button,
             const gchar *info )
{
  GtkWidget *section;

  section = gtk_flow_box_new ();
  gtk_widget_set_halign (section, GTK_ALIGN_START);

  gtk_flow_box_set_selection_mode (GTK_FLOW_BOX (section), GTK_SELECTION_NONE);
  gtk_flow_box_set_min_children_per_line (GTK_FLOW_BOX (section), 2);
  gtk_flow_box_set_max_children_per_line (GTK_FLOW_BOX (section), 20);
  gtk_widget_set_tooltip_text (button, info);
  gtk_container_add (GTK_CONTAINER (section), button);

  return section;
}

void CreatBeginButton(Junqi *pJunqi)
{
    //开始按钮
    GtkWidget *button_box = gtk_event_box_new();
    GdkPixbuf *pixbuf = gdk_pixbuf_new_from_file_at_scale(
    		"./res/begin.gif", 100,100,1,NULL);
    GtkWidget *image = gtk_image_new_from_pixbuf(pixbuf);
    gtk_container_add(GTK_CONTAINER(button_box),image);
    pJunqi->begin_button = image;
    g_signal_connect(button_box,"button-press-event",G_CALLBACK(begin_button),pJunqi);
    gtk_fixed_put(GTK_FIXED(pJunqi->fixed), button_box, 580,520);
    g_object_unref (pixbuf);
    gtk_widget_show(button_box);
    gtk_widget_show(image);
    //立即出招按钮
    GtkWidget *start_button = gtk_event_box_new();
    GtkWidget *section;
	pixbuf = gdk_pixbuf_new_from_file_at_scale(
			"./res/start2.png", 40,40,1,NULL);
	image = gtk_image_new_from_pixbuf(pixbuf);
	pJunqi->start_button = image;
	gtk_container_add(GTK_CONTAINER(start_button),image);
	gtk_widget_show(start_button);
	g_signal_connect(start_button,"button-press-event",G_CALLBACK(send_go),pJunqi);
	section = add_section(start_button,"立即出招");
	gtk_widget_show(section);
	gtk_fixed_put(GTK_FIXED(pJunqi->fixed), section, 580,520);
	g_object_unref (pixbuf);

}

void CreatAPPIcon(Junqi *pJunqi)
{
	GdkPixbuf *pixbuf;
	GtkWidget *window;

	window = pJunqi->window;
	pixbuf = gdk_pixbuf_new_from_file("./res/JunQi.ico", NULL);
	gtk_window_set_icon(GTK_WINDOW(window), pixbuf);
}

void SetTimeStr(char *label_str, int time)
{
	snprintf(
			label_str,
			100,
			"<span foreground=\"#FFFF00\" font='16'>%02d</span>",
			time);
}

void CreatStepSlider(Junqi *pJunqi)
{
	GtkWidget *fixed;
	GtkWidget *image;
	GtkWidget *button;
	int length = 200;

	fixed = gtk_fixed_new ();
	gtk_widget_set_size_request(fixed, length,100);
	gtk_fixed_put(GTK_FIXED(pJunqi->fixed), fixed, 500, 500);
	pJunqi->step_fixed = fixed;
	//gtk_widget_show (fixed);

	image = get_image_from_name("./res/background.PNG",0, 0, length, 100);
	gtk_fixed_put(GTK_FIXED(fixed), image, 0, 0);
	gtk_widget_show (image);

    GtkWidget *label, *slider;
    label = gtk_label_new(NULL);
	gtk_label_set_markup(GTK_LABEL(label),"<span font='14'>步 数 </span>");
	gtk_fixed_put(GTK_FIXED(fixed), label, 10, 10);
	gtk_widget_show (label);

	image = get_image_from_name("./res/step_button.PNG",4, 6, 27, 27);
	button = gtk_button_new();
	gtk_button_set_image(GTK_BUTTON(button), image);
	gtk_fixed_put(GTK_FIXED(fixed), button, length/2, 10);
	gtk_widget_show (button);
	g_signal_connect (button, "clicked", G_CALLBACK (step_cb), "left");

	image = get_image_from_name("./res/step_button.PNG",65, 6, 27, 27);
	button = gtk_button_new();
	gtk_button_set_image(GTK_BUTTON(button), image);
	gtk_fixed_put(GTK_FIXED(fixed), button, length*3/4, 10);
	gtk_widget_show (button);
	g_signal_connect (button, "clicked", G_CALLBACK (step_cb), "right");

    GtkAdjustment *adj;
    adj = gtk_adjustment_new (0, 0, 100, 1, 1, 0);
    slider = gtk_scale_new (GTK_ORIENTATION_HORIZONTAL, adj);
    pJunqi->slider_adj = adj;
    g_signal_connect (adj, "value-changed",
                      G_CALLBACK (on_slider_value_change),
                      pJunqi);
    gtk_widget_set_size_request(slider, length,20);
    gtk_scale_set_digits(GTK_SCALE (slider), 0);
    gtk_fixed_put(GTK_FIXED(fixed), slider, 0, 50);
    gtk_widget_show (slider);

	return ;
}

void OpenBoard(GtkWidget *window)
{
	GtkWidget* draw_area = gtk_drawing_area_new();
	GtkWidget *vbox, *hbox, *vbox2;
	Junqi *pJunqi = JunqiOpen();
	if(pJunqi == NULL)
	{
		g_warning("JunQi initialization failed");
		gtk_main_quit();
		return;
	}
	LoadConfig(pJunqi);
	pJunqi->window = window;
	gJunqi = pJunqi;
	g_signal_connect(window, "destroy", G_CALLBACK(stop_audio_on_destroy), pJunqi);
    hbox = gtk_box_new (GTK_ORIENTATION_HORIZONTAL, 0);
    gtk_container_add (GTK_CONTAINER (window), hbox);
    vbox = gtk_box_new (GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start (GTK_BOX (hbox), vbox, TRUE, TRUE, 0);
    vbox2 = gtk_box_new (GTK_ORIENTATION_VERTICAL, 0);
    gtk_box_pack_start (GTK_BOX (hbox), vbox2, FALSE, TRUE, 0);
    //新建菜单
    set_menu(vbox);

    GtkWidget *event_box = gtk_event_box_new();
    gtk_box_pack_start (GTK_BOX (vbox), event_box, TRUE, TRUE, 0);
    //fixed用来存放各种容器，把fixed放在事件盒子里，来处理鼠标事件
    GtkWidget *fixed = gtk_fixed_new();
    pJunqi->fixed = fixed;
    gtk_container_add(GTK_CONTAINER(event_box),fixed);
    gtk_widget_set_size_request(draw_area, 733,688);
    gtk_fixed_put(GTK_FIXED(fixed), draw_area, 0, 0);
    background = gdk_pixbuf_new_from_file("./res/1024board.bmp", NULL);
    g_signal_connect (draw_area, "draw",G_CALLBACK (draw_cb), NULL);

    gtk_widget_show_all(window);

    pJunqi->pTimeLabel = gtk_label_new(NULL);
    CreatBeginButton(pJunqi);
    SetButton(window,pJunqi);
    if(pJunqi->eShowMode != SHOW_MODE_BRIGHT)
    {
        g_timeout_add(1000, (GSourceFunc)time_event, pJunqi);
    }
    pthread_t tidp;
    if(pthread_create(&tidp, NULL, sound_thread, pJunqi) != 0)
    {
		g_warning("unable to start sound thread; continuing without audio");
		pJunqi->bMute = 1;
    }
    else
    {
		pthread_detach(tidp);
    }
    CreatStepSlider(pJunqi);
    CreatBoardChess(window, pJunqi);
    CreatCommThread(pJunqi);

}
